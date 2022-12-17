import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config


def main(args, left_argv):
    device = 'cpu'

    config = Config(args.config)
    config.argv_update(left_argv)

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
        ape_last_block=config.ape_last_block
    )
    prompt = "<s_tableocr>"
    prompt_tensors = model.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    print(prompt_tensors)
    print(prompt_tensors.dtype)
    print(type(prompt_tensors))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--output_path', default='efbl0_quantized.torchscript', type=str)
    parser.add_argument('--use_optimizer', default=False, action='store_true')
    parser.add_argument('--use_script', default=False, action='store_true')

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
