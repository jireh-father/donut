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
    model = model.decoder
    max_len = 3000
    input_ids = torch.tensor([[model.tokenizer.get_vocab()["<s_tableocr>"]] + [model.pad_token_id] * (max_len - 1)])
    # input_ids = model.tokenizer("<s_tableocr>", add_special_tokens=False, return_tensors="pt")["input_ids"]

    model.forward = model.inference_decode_one_step_for_android
    return_index = torch.tensor([0])


    example = torch.rand(1, 1200, 1024)
    if device == 'cpu':
        model.to(device)
        # model.to(torch.bfloat16)
        example.to(device)
        input_ids.to(device)
        return_index = return_index.to(device)
        # example = example.to(torch.bfloat16)
    else:
        model.half()
        model.to(device)
        example = example.half()
        example = example.to(device)
        input_ids = input_ids.half()
        input_ids = input_ids.to(device)
        return_index = return_index.half()
        return_index = return_index.to(device)
    model.eval()

    if args.use_script:
        traced_script_module = torch.jit.script(model)
    else:
        ret = model(input_ids, example, return_index)
        print("ret", ret)
        print(ret.shape)
        traced_script_module = torch.jit.trace(model, (input_ids, example, return_index))

    if args.use_optimizer:
        traced_script_module = optimize_for_mobile(traced_script_module)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced_script_module.save(args.output_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='./config/train_swinv2_realworld_synth_remove_img_tag_tokenizer_from_scratch_1280x1280_for_test_in_pc.yaml')
    parser.add_argument('--output_path', default='D:\\result\\tableocr\\android/decoder_1280x1280.ptl', type=str)
    parser.add_argument('--use_optimizer', default=False, action='store_true')
    parser.add_argument('--use_script', default=False, action='store_true')

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
