import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config

model_classifier_map = {
    'alexnet': ['classifier', 6],
    'vgg': ['classifier', 6],
    'mobilenet_v3': ['classifier', 3],
    'mobilenet_v2': ['classifier', 1],
    'mnasnet': ['classifier', 6],
    'resnet': ['fc'],
    'inception': ['fc'],
    'googlenet': ['fc'],
    'shufflenet': ['fc'],
    'densenet': ['classifier'],
    'resnext': ['fc'],
    'wide_resnet': ['fc'],
    'efficientnet': ['_fc'],
    'bagnet': ['fc'],
    'rexnet': ['output', 1],
}


def init_model(model_name, num_classes, model_path=None):

    if model_name.startswith("efficientnet_lite"):
        import timm
        from torch.quantization import QuantStub, DeQuantStub
        model = timm.create_model(model_name, pretrained=True)

        class QuantWrapper(nn.Module):
            def __init__(self, base_model):
                super(QuantWrapper, self).__init__()
                self.base_model = base_model
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.base_model(x)
                x = self.dequant(x)
                return x

        quant_model = QuantWrapper(model)
        return quant_model
    if model_name.startswith("efficientnet-lite"):
        from efficientnet_lite_pytorch import EfficientNet
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
        return model

    if model_name.startswith("efficientnet"):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        return model

    from torchvision import models
    pretrained = True if model_path is None else False
    for m_key in model_classifier_map:
        if m_key in model_name:
            model_fn = getattr(models, model_name)
            cls_layers = model_classifier_map[m_key]

            if model_name.startswith("inception"):
                # input_size = 299
                model = model_fn(aux_logits=False, pretrained=pretrained)
            else:
                # input_size = 224
                model = model_fn(pretrained=pretrained)

            if len(cls_layers) == 1:
                in_features = getattr(model, cls_layers[0]).in_features
                setattr(model, cls_layers[0], nn.Linear(in_features, num_classes))
            else:
                classifier = getattr(model, cls_layers[0])
                in_features = classifier[cls_layers[1]].in_features
                classifier[cls_layers[1]] = nn.Linear(in_features, num_classes)
            return model


def main(args, left_argv):
    device = 'cpu'

    config = Config(args.config)
    config.argv_update(left_argv)

    task_start_tokens = [f"<s_{config.task_name}>"]
    prompt_end_tokens = [f"<s_{config.task_name}>"]

    model = DonutModel.from_pretrained(
        config.pretrained_model_name_or_path,
        input_size=config.input_size,
        max_length=config.max_length,
        align_long_axis=config.align_long_axis,
        ignore_mismatched_sizes=True,
        # use_fast_tokenizer=config.use_fast_tokenizer,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        # vision_model_name=config.vision_model_name,
        bart_prtrained_path=config.bart_prtrained_path,
        special_tokens=task_start_tokens + prompt_end_tokens,
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

    if device == 'cpu':
        model.encoder.to(torch.bfloat16)
        # todo: 원래 없던 소스임. 확인 필요
        model.to(device)
    else:
        model.half()
        model.to("cuda")
    model.eval()

    if args.use_script:
        traced_script_module = torch.jit.script(model)
    else:
        example = torch.rand(1, 3, config.input_size, config.input_size)
        ret = model(example)
        print(ret, ret.shape)
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
