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
    os.makedirs(args.output_dir, exist_ok=True)
    vocab = model.decoder.tokenizer.get_vocab()
    json.dump(vocab, open(os.path.join(args.output_dir, "word_to_idx.json"), "w+"))
    inv_vocab = {"{}".format(v): k for k, v in vocab.items()}
    json.dump(inv_vocab, open(os.path.join(args.output_dir, "idx_to_word.json"), "w+"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='D:\\result\\tableocr\\android\\tokenizer_dict_json')
    parser.add_argument("--config", type=str, default='./config/train_swinv2_realworld_synth_remove_img_tag_tokenizer_from_scratch_1280x1280_for_test_in_pc.yaml')
    # 8666/9115
    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    # config.argv_update(left_argv)

    test(args, config)
