"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutModelForTableOcrTest, DonutConfig
import teds
from sconf import Config


def test(args, config):
    # pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    if args.task_name == "tableocr":

        pretrained_model = DonutModel.from_pretrained(
            args.pretrained_model_name_or_path,
            input_size=config.input_size,
            max_length=config.max_length,
            align_long_axis=config.align_long_axis,
            ignore_mismatched_sizes=True,
        )
    else:
        pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")
    else:
        pretrained_model.encoder.to(torch.bfloat16)

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    output_list = []
    accs = []
    teds_structure_results = []

    if args.task_name == "tableocr":
        teds_metric_stru = teds.TEDS(True, n_jobs=args.num_processes)
        teds_metric = teds.TEDS(n_jobs=args.num_processes)

        dataset = load_dataset(args.dataset_name_or_path, data_files='metadata.jsonl')['train']
    else:
        dataset = load_dataset(args.dataset_name_or_path, split=args.split)

    # for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    gt_list = []
    pred_list = []
    for idx, sample in enumerate(dataset):
        ground_truth = json.loads(sample["ground_truth"])
        print("###", sample["file_name"], "{}/{}".format(idx, len(dataset)))
        im = Image.open(os.path.join(args.dataset_name_or_path, sample["file_name"]))
        output = pretrained_model.inference(im, prompt=f"<s_{args.task_name}>")["predictions"][0]

        gt = ground_truth["gt_parse"]["text_sequence"]
        output = teds.postprocess_html_tag(output['text_sequence'])
        gt = teds.postprocess_html_tag(gt)

        output_list.append(output)

        if args.num_processes > 1:
            gt_list.append(gt)
            pred_list.append(output)
            if len(gt_list) == args.num_processes:
                scores = teds_metric.batch(pred_list, gt_list)
                stru_scores = teds_metric_stru.batch(pred_list, gt_list)
                accs += scores
                teds_structure_results += stru_scores
                if args.verbose:
                    for j, gt in enumerate(gt_list):
                        output = pred_list[j]
                        print("===== true")
                        print(gt)
                        print("===== pred")
                        print(output)
                        print("===== teds all", scores[j])
                        print("===== teds only structure", stru_scores[j])

                gt_list = []
                pred_list = []
        else:
            score = teds_metric.evaluate(output, gt)
            teds_structure_score = teds_metric_stru.evaluate(output, gt)
            if args.verbose:
                print("===== true")
                print(gt)
                print("===== pred")
                print(output)
                print("===== teds all", score)
                print("===== teds only structure", teds_structure_score)

            accs.append(score)
            teds_structure_results.append(teds_structure_score)

    if args.task_name == "tableocr":
        scores = {"teds": accs, "mean_teds": np.mean(accs), "teds_structure": teds_structure_results,
                  "mean_teds_structure": np.mean(teds_structure_results)}
        print("teds all", scores['mean_teds'], f"length : {len(accs)}")
        print("teds only structure", scores['mean_teds_structure'], f"length : {len(accs)}")
    else:
        scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
        print(scores, f"length : {len(accs)}")

    if args.save_path:
        scores["predictions"] = output_list
        save_json(args.save_path, scores)

    return output_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    config.argv_update(left_argv)

    predicts = test(args, config)
