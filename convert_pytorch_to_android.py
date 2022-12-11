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

    model.forward = model.inference_one

    example = torch.rand(1, 3, config.input_size[0], config.input_size[1])
    if device == 'cpu':
        model.encoder.to(torch.bfloat16)
        # todo: 원래 없던 소스임. 확인 필요
        model.to(device)
        example = example.to(torch.bfloat16)
        example = example.to(device)
    else:
        model.half()
        model.to(device)
        example = example.half()
        example = example.to(device)
    model.eval()

    # image_tensors = self.encoder.prepare_input(image).unsqueeze(0)
    #         if self.device.type == "cuda":  # half is not compatible in cpu implementation.
    #             image_tensors = image_tensors.half()
    #             image_tensors = image_tensors.to(self.device)
    #         else:
    #             image_tensors = image_tensors.to(torch.bfloat16)

    if args.use_script:
        traced_script_module = torch.jit.script(model)
    else:
        ret = model(example)
        print("ret", ret)
        traced_script_module = torch.jit.trace(model, example)

    if args.use_optimizer:
        traced_script_module = optimize_for_mobile(traced_script_module)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced_script_module.save(args.output_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--output_path', default='efbl0_quantized.torchscript', type=str)
    parser.add_argument('--use_optimizer', default=True, action='store_true')
    parser.add_argument('--use_script', default=False, action='store_true')

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
