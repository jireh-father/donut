import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config
import time
import glob
from PIL import Image


def main(args, left_argv):
    config = Config(args.config)
    config.argv_update(left_argv)

    real_model = DonutModel.from_pretrained(
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
        ape_last_block=config.ape_last_block
    )
    print("loaded pytorch model")
    model = torch.jit.load(args.model_path)
    print("loaded torchscript model")
    config = Config(args.config)
    config.argv_update(left_argv)
    device = 'cpu'

    image_files = glob.glob(args.images_pattern)
    im = Image.open(image_files[0])
    if im.mode != "RGB":
        im = im.convert("RGB")
    input_tensors = []
    input_tensor = real_model.encoder.prepare_input(im, random_padding=False)
    input_tensors.append(input_tensor)
    example = torch.stack(input_tensors, dim=0).to(device)
    print("started torchscript inference")
    start = time.time()
    ret = model(example)
    print(time.time() - start)
    print(ret)
    print(len(ret[0]))
    print("ended torchscript inference")

    for image_file in image_files:
        im = Image.open(image_file)
        if im.mode != "RGB":
            im = im.convert("RGB")
        input_tensors = []
        input_tensor = real_model.encoder.prepare_input(im, random_padding=False)
        input_tensors.append(input_tensor)
        example = torch.stack(input_tensors, dim=0).to(device)
        print("started torchscript inference")
        start = time.time()
        ret = model(example)
        print(time.time() - start)
        print(ret)

    #
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_pattern", type=str, default='D:\dataset\\table_ocr\\test_sample_en_one/*')
    parser.add_argument("--config", type=str, default="./config/train_swinv2_realworld_synth_for_test_in_pc.yaml")
    parser.add_argument('--model_path', default='D:\\result\\tableocr\\android/e2e_model.pth', type=str)

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
