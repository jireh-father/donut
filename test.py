"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import pickle
import traceback
import glob
import argparse
import json
import os
import re
import shutil
from pathlib import Path
import re
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
import teds as T
from sconf import Config


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


multiple_space_re = re.compile(r'[ ]{2,}')


def convert(text):
    text = " ".join([remove_html_tags(token) for token in text.split("</td>")])
    text = multiple_space_re.sub(' ', text)
    return "<tr><td>{}</td></tr>".format(text.strip())


def test(args, config):
    model = DonutModel.from_pretrained(
        args.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        ignore_mismatched_sizes=True,
        use_fast_tokenizer=config.use_fast_tokenizer,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        vision_model_name=config.vision_model_name,
        bart_prtrained_path=config.bart_prtrained_path,
        special_tokens=['<s_tableocr>'],
        swin_pretrained_path=config.swin_pretrained_path,
        window_size=config.window_size,
        swin_model_size=config.swin_model_size,
        ape=config.ape
    )

    if torch.cuda.is_available():
        model.half()
        model.to("cuda")
    else:
        model.encoder.to(torch.bfloat16)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    teds_metric_struct = T.TEDS(True, n_jobs=args.num_processes)
    teds_metric = T.TEDS(n_jobs=args.num_processes)
    dataset = load_dataset(args.dataset_name_or_path, data_files='metadata.jsonl')['train']

    tag_re = re.compile(r'<[^>]+>')
    gt_list = []
    pred_list = []
    file_list = []

    total_teds_all = []
    total_teds_struct = []
    # total_teds_content = []

    error_data = []

    result_path = os.path.join(args.output_dir, "results.jsonl")
    with open(result_path, "w+", encoding="utf-8") as output:
        for idx, sample in enumerate(dataset):
            if args.start_index and args.start_index > idx:
                if idx % 10 == 0:
                    print("skip", idx)
                continue
            if args.test_cnt and args.test_cnt < idx:
                break
            file_name = sample["file_name"]
            print("###", file_name, "{}/{}".format(idx, len(dataset)))
            sample_data = json.loads(sample["ground_truth"])
            im = Image.open(os.path.join(args.dataset_name_or_path, file_name))
            width, height = im.size
            pred = model.inference(im, prompt=f"<s_tableocr>")["predictions"][0]
            gt = T.postprocess_html_tag(sample_data["gt_parse"]["text_sequence"])
            pred = T.postprocess_html_tag(pred['text_sequence'])

            file_list.append(file_name)
            gt_list.append(gt)
            pred_list.append(pred)
            if len(gt_list) == args.num_processes:
                teds_all_list = teds_metric.batch(pred_list, gt_list)

                teds_struct_list = teds_metric_struct.batch(pred_list, gt_list)
                pred_content_list = [convert(pred) for pred in pred_list]
                gt_content_list = [convert(gt) for gt in gt_list]
                # teds_content_list = teds_metric.batch(pred_content_list, gt_content_list)
                total_teds_all += teds_all_list
                total_teds_struct += teds_struct_list
                # total_teds_content += teds_content_list

                for j, gt in enumerate(gt_list):
                    teds_all = teds_all_list[j]
                    teds_struct = teds_struct_list[j]
                    # teds_content = teds_content_list[j]
                    pred = pred_list[j]

                    item = {
                        "file_name": file_list[j],
                        "gt": gt,
                        "pred": pred,
                        "teds_all": teds_all,
                        "teds_struct": teds_struct,
                        # "teds_content": teds_content,
                        "image_width": width,
                        "image_height": height
                    }
                    try:
                        output.write("{}\n".format(json.dumps(item)))
                    except:
                        item['index'] = idx
                        error_data.append(item)
                        traceback.print_exc()

                    if args.verbose:
                        print("")
                        print("#####", file_list[j])
                        print("===== gt")
                        print(gt)
                        print("===== pred")
                        print(pred)
                        print("===== gt contents")
                        print(gt_content_list[j])
                        print("===== pred contents")
                        print(pred_content_list[j])
                        print("===== gt structure")
                        print("".join(tag_re.findall(gt)))
                        print("===== pred structure")
                        print("".join(tag_re.findall(pred)))
                        print("===== teds all", teds_all)
                        print("===== teds structure", teds_struct)
                        # print("===== teds content", teds_content)
                gt_list = []
                pred_list = []
                file_list = []

    total_teds_mean = {
        "teds_all": np.mean(total_teds_all),
        "teds_structure": np.mean(total_teds_struct),
        # "teds_content": np.mean(total_teds_content)
    }

    if error_data:
        pickle.dump(error_data, open(os.path.join(args.output_dir, "errors.pickle"), "wb+"))
        print("errors", len(error_data))

    print(total_teds_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--test_cnt", type=int, default=None)
    # 8666/9115
    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    config.argv_update(left_argv)

    test(args, config)
