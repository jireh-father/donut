"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import glob
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
import teds as T
from sconf import Config


def test(args, config):
    model = DonutModel.from_pretrained(
        args.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        ignore_mismatched_sizes=True,
    )

    if torch.cuda.is_available():
        model.half()
        model.to("cuda")
    else:
        model.encoder.to(torch.bfloat16)
    model.eval()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    teds_metric_struct = T.TEDS(True, n_jobs=args.num_processes)
    teds_metric = T.TEDS(n_jobs=args.num_processes)

    dataset = load_dataset(args.dataset_name_or_path, data_files='metadata.jsonl')['train']

    total_result = {}

    score_dict = {
        "total_pred_list": [],
        "total_teds": [],
        "total_struct_teds": [],

        "simple_teds": [],
        "simple_struct_teds": [],
        "complex_teds": [],
        "complex_struct_teds": [],

        "teds_by_all_token_count": {},
        "teds_struct_by_all_token_count": {},

        "teds_by_content_token_count": {},
        "teds_struct_tag_by_content_token_count": {},

        "teds_by_tag_token_count": {},
        "teds_struct_tag_by_token_count": {},
        # 행열갯수
        # 이미지사이즈별
        # 최대 rowspan 갯수별
        # 최대 colspan 갯수별
        # cell 평균 크기별(width, height)
        # row 및 col 크기별
    }

    gt_list = []
    pred_list = []
    file_list = []

    for idx, sample in enumerate(dataset):
        ground_truth = json.loads(sample["ground_truth"])
        print("###", sample["file_name"], "{}/{}".format(idx, len(dataset)))
        im = Image.open(os.path.join(args.dataset_name_or_path, sample["file_name"]))
        output = model.inference(im, prompt=f"<s_{args.task_name}>")["predictions"][0]

        gt = ground_truth["gt_parse"]["text_sequence"]
        output = T.postprocess_html_tag(output['text_sequence'])
        gt = T.postprocess_html_tag(gt)

        total_pred_list.append(output)
        file_list.append(sample["file_name"])

        gt_list.append(gt)
        pred_list.append(output)
        if len(gt_list) == args.num_processes:
            scores = teds_metric.batch(pred_list, gt_list)
            stru_scores = teds_metric_struct.batch(pred_list, gt_list)
            total_teds += scores
            total_struct_teds += stru_scores

            for j, gt in enumerate(gt_list):
                cur_score = scores[j]
                cur_struct_score = stru_scores[j]
                output = pred_list[j]

                token_len = len(model.decoder.tokenizer.encode(gt))

                is_simple = False
                if "rowspan" in gt or "colspan" in gt:
                    complex_teds.append(cur_score)
                    complex_struct_teds.append(cur_struct_score)
                    is_simple = True
                else:
                    simple_teds.append(cur_score)
                    simple_struct_teds.append(cur_struct_score)

                if args.verbose:
                    print("#####", file_list[j])
                    print("===== true")
                    print(gt)
                    print("===== pred")
                    print(output)
                    print("===== teds all", scores[j])
                    print("===== teds only structure", stru_scores[j])
                    print("===== is_simple", is_simple)
                if args.save_images and args.output_dir:
                    file_name = file_list[j]
                    image_path = os.path.join(args.dataset_name_or_path, file_name)
                    im_output_path = os.path.join(args.output_dir, "result_images_all",
                                                  "10" if scores[j] == 1.0 else str(scores[j])[2])
                    os.makedirs(im_output_path, exist_ok=True)
                    shutil.copy(image_path, im_output_path)

                    im_output_path = os.path.join(args.output_dir, "result_images_structure",
                                                  "10" if stru_scores[j] == 1.0 else str(stru_scores[j])[2])
                    os.makedirs(im_output_path, exist_ok=True)
                    shutil.copy(image_path, im_output_path)
            gt_list = []
            pred_list = []
            file_list = []

    # simple_teds = []
    # simple_struct_teds = []
    # complex_teds = []
    # complex_struct_teds = []

    scores = {
        "mean_teds": np.mean(total_teds),
        "teds_structure": total_struct_teds,
        "mean_teds_structure": np.mean(total_struct_teds)
    }
    print("teds all", scores['mean_teds'], f"length : {len(total_teds)}")
    print("teds only structure", scores['mean_teds_structure'], f"length : {len(total_teds)}")
    print("teds all", scores['mean_teds'], f"length : {len(total_teds)}")
    print("teds only structure", scores['mean_teds_structure'], f"length : {len(total_teds)}")
    print("teds all", scores['mean_teds'], f"length : {len(total_teds)}")
    print("teds only structure", scores['mean_teds_structure'], f"length : {len(total_teds)}")

    if args.output_dir:
        scores["predictions"] = total_pred_list
        save_json(os.path.join(args.output_dir, "result.json"), scores)
        for score_dir in glob.glob(os.path.join(args.output_dir, "result_images_all", "*")):
            if not os.path.isdir(score_dir):
                continue
            print("teds all", os.path.basename(score_dir), len(glob.glob(os.path.join(score_dir, "*"))))
        for score_dir in glob.glob(os.path.join(args.output_dir, "result_images_structure", "*")):
            if not os.path.isdir(score_dir):
                continue
            print("teds stru", os.path.basename(score_dir), len(glob.glob(os.path.join(score_dir, "*"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--save_images", action='store_true', default=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    config.argv_update(left_argv)

    test(args, config)
