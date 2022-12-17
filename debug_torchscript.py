import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config
import time


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = nn.Linear(2, 512, bias=True)
        self.b = nn.ReLU()
        self.c = nn.Linear(512, 8, bias=False)
        # self.cpb_mlp = nn.Sequential(
        #     nn.Linear(2, 512, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 8, bias=False)
        # )

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        return x


def main(args, left_argv):
    model = Model()
    config = Config(args.config)
    config.argv_update(left_argv)

    example = torch.rand(1, 2)
    model = model.to(torch.bfloat16)
    example = example.to(torch.bfloat16)
    model.eval()

    ret = model(example)
    print("ret", ret)
    traced_script_module = torch.jit.trace(model, example)

    # traced_script_module = optimize_for_mobile(traced_script_module)

    ret = traced_script_module(example)
    print(ret)

    # print(time.time() - start)
    # print(ret)
    #
    # print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./config/train_swinv2_realworld_synth_remove_img_tag_resume.yaml")
    parser.add_argument('--model_path', default='D:\\result\\tableocr\\android/donut.ptl', type=str)

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
