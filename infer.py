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

    image_files = os.path.join(args.image_dir, "*.jpg")

    for idx, im_path in enumerate(image_files):
        if args.start_index and args.start_index > idx:
            if idx % 10 == 0:
                print("skip", idx)
            continue
        if args.test_cnt and args.test_cnt < idx:
            break

        im = Image.open(im_path)
        with torch.set_grad_enabled(False):
            pred = model.inference(im, prompt=f"<s_tableocr>")["predictions"][0]
        pred = T.postprocess_html_tag(pred['text_sequence'])

        output_path = os.path.join(args.output_dir, os.path.basename(im_path)) + ".html"
        with open(output_path, "w+", encoding="utf-8") as outputf:
            outputf.write(pred)
        print("###", os.path.basename(im_path), "{}/{}".format(idx, len(image_files)))
        print(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='D:\\result\\tableocr\models\\tokenizer_swinv2_max4k_fs')
    parser.add_argument("--image_dir", type=str, default='D:\dataset\\table_ocr\\test_samples_en')
    parser.add_argument("--output_dir", type=str, default='D:\\result\\tableocr\infer_results')
    parser.add_argument("--config", type=str, required=True, default='./config/train_table_ocr_v2_swinv2_max4k.yaml')
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--test_cnt", type=int, default=None)
    # 8666/9115
    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    config.argv_update(left_argv)

    test(args, config)
