import torch
import torch.nn as nn
import argparse
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from donut import DonutModel, JSONParseEvaluator, load_json, save_json, DonutConfig
from sconf import Config
import time

def main(args, left_argv):
    model = torch.jit.load(args.model_path)
    config = Config(args.config)
    config.argv_update(left_argv)
    start = time.time()
    ret = model(torch.rand(1, 3, config.input_size[0], config.input_size[1]).to(torch.bfloat16))
    print(time.time() - start)
    print(ret)
    #
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/train_swinv2_realworld_synth_remove_img_tag_resume.yaml")
    parser.add_argument('--model_path', default='D:\\result\\tableocr\\android/donut.ptl', type=str)

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
