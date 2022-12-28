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
import imgkit
import time

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


multiple_space_re = re.compile(r'[ ]{2,}')
html_template = """
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body>
            {}
        </body>
        </html>
        """

def convert(text):
    text = " ".join([remove_html_tags(token) for token in text.split("</td>")])
    text = multiple_space_re.sub(' ', text)
    return "<tr><td>{}</td></tr>".format(text.strip())


def test(args, config):
    model = DonutModel.from_pretrained(
        config.pretrained_model_name_or_path,
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
        ape=config.ape,
        swin_name_or_path=config.swin_name_or_path,
        encoder_layer=config.swin_encoder_layer,
        d_model=config.d_model,
        swin_depth_last_block=config.swin_depth_last_block,
        swin_num_heads_last_block=config.swin_num_heads_last_block,
        swin_drop_path_rate_last_block=config.swin_drop_path_rate_last_block,
        swin_init_values_last_block=config.swin_init_values_last_block,
        ape_last_block=config.ape_last_block,
        local_files_only=True
    )


    if torch.cuda.is_available():
        model.half()
        model.to("cuda")
        model.decoder.device = "cuda"
        print("cuda")
    else:
        model.encoder.to(torch.bfloat16)
        model.decoder.device = "cpu"
        print("cpu")
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = glob.glob(args.images_pattern)

    vocab = model.decoder.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    for start_idx in range(0, len(image_files), args.batch_size):
        print(start_idx, len(image_files))
        batch_images = image_files[start_idx:start_idx + args.batch_size]
        input_tensors = []
        for image_path in batch_images:
            im = Image.open(image_path)
            if im.mode != "RGB":
                im = im.convert("RGB")
            input_tensor = model.encoder.prepare_input(im, random_padding=False)
            input_tensors.append(input_tensor)
        input_tensors = torch.stack(input_tensors, dim=0)
        print("input_tensors", input_tensors.shape)

        if torch.cuda.is_available():
            input_tensors = input_tensors.half()
            input_tensors = input_tensors.to('cuda')
        else:
            input_tensors = input_tensors.to('cpu')

        # preds = model.inference(image_tensors=input_tensors, prompt=f"<s_tableocr>")["predictions"]
        start = time.time()
        # preds = model.inference_direct(image_tensors=input_tensors, prompt=f"<s_tableocr>")
        preds = model.inference_for_android(image_tensors=input_tensors)
        print("inference time", time.time() - start)
        print("preds", preds.shape)
        for i, pred in enumerate(preds):
            print(pred)
            pred = [inv_vocab.get(int(p), "") for p in pred]
            print(pred)
            # pred = model.decoder.tokenizer.decode(pred)
            # if pred.startswith("<s_tableocr> "):
            #     pred = pred[len("<s_tableocr> "):]
            # if pred.endswith("</s>"):
            #     pred = pred[:-len("</s>")]
            pred = "".join(pred[1:-1]).replace("▁", " ")
            print(pred)
            pred = T.postprocess_html_tag(re.sub(r"(?:(?<=>) | (?=</s_))", "", pred))
            print(pred)
            shutil.copy(batch_images[i], args.output_dir)
            only_file_name = os.path.splitext(os.path.basename(batch_images[i]))[0]
            print(pred)
            output_html = html_template.format(pred.replace("<table>", '<table border="1">'))
            html_path = os.path.join(args.output_dir, only_file_name + ".html")
            with open(html_path, "w+", encoding='utf-8') as fp:
                fp.write(output_html)
            html_jpg_path = os.path.join(args.output_dir, only_file_name + "_html.jpg")
            if args.use_imgkit:
                imgkit.from_file(html_path, html_jpg_path, options={"xvfb": ""})

        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_pattern", type=str, default='D:\dataset\\table_ocr\\test_sample_en_one2/*')
    parser.add_argument("--output_dir", type=str, default='D:\\result\\tableocr\infer_results_one')
    parser.add_argument("--config", type=str, default='./config/train_swinv2_realworld_synth_for_test_in_pc.yaml')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_imgkit", action='store_true', default=False)
    # 8666/9115
    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    # config.argv_update(left_argv)

    test(args, config)
