import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config
import time

def main(args, left_argv):
    config = Config(args.config)
    config.argv_update(left_argv)

    ori_model = DonutModel.from_pretrained(
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
    ori_model = ori_model.decoder
    # input_ids = ori_model.tokenizer("<s_tableocr>", add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.tensor([[ori_model.tokenizer.get_vocab()["<s_tableocr>"]] + [ori_model.pad_token_id] * 100])
    model = torch.jit.load(args.model_path)
    config = Config(args.config)
    config.argv_update(left_argv)

    device = 'cpu'
    input_ids = input_ids.to(device)
    hidden_state = torch.rand(1, 1200, 1024).to(device)
    hidden_state = hidden_state.to(device)
    start = time.time()
    ret = model(input_ids, hidden_state)
    print(time.time() - start)
    print(ret)
    print(ret.shape)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/train_swinv2_realworld_synth_for_test_in_pc.yaml')
    parser.add_argument('--model_path', default='D:\\result\\tableocr\\android/only_decoder.pth', type=str)

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
