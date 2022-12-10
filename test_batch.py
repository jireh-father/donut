"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import pickle
import traceback
import argparse
import json
import os
import re
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from donut import DonutModel
import teds as T
from sconf import Config
from collections import defaultdict


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


multiple_space_re = re.compile(r'[ ]{2,}')


def convert(text):
    text = " ".join([remove_html_tags(token) for token in text.split("</td>")])
    text = multiple_space_re.sub(' ', text)
    return "<tr><td>{}</td></tr>".format(text.strip())


def test(args, config):
    # model = DonutModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     input_size=config.input_size,
    #     max_length=config.max_length,
    #     align_long_axis=config.align_long_axis,
    #     ignore_mismatched_sizes=True,
    #     use_fast_tokenizer=config.use_fast_tokenizer,
    #     tokenizer_name_or_path=config.tokenizer_name_or_path,
    #     vision_model_name=config.vision_model_name,
    #     bart_prtrained_path=config.bart_prtrained_path,
    #     special_tokens=['<s_tableocr>'],
    #     swin_pretrained_path=config.swin_pretrained_path,
    #     window_size=config.window_size,
    #     swin_model_size=config.swin_model_size,
    #     ape=config.ape
    # )
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

    if torch.cuda.is_available():
        model.half()
        model.to("cuda")
    else:
        model.encoder.to(torch.bfloat16)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    teds_metric_struct = T.TEDS(True, n_jobs=config.num_workers)
    teds_metric = T.TEDS(n_jobs=config.num_workers)

    tag_re = re.compile(r'<[^>]+>')
    gt_list = []
    pred_list = []
    file_list = []

    total_teds_all = []
    total_teds_struct = []
    dataset_teds_all = defaultdict(list)
    dataset_teds_struct = defaultdict(list)
    # total_teds_content = []

    error_data = []

    result_path = os.path.join(args.output_dir, "results.jsonl")
    dataset_path_list = config.dataset_name_or_paths
    with open(result_path, "w+", encoding="utf-8") as output:
        for dataset_idx, dataset_name_or_path in enumerate(dataset_path_list):
            dataset_name = os.path.basename(dataset_name_or_path)
            dataset = load_dataset(dataset_name_or_path, data_files={"validation": "validation/metadata.jsonl"})[
                "validation"]

            dataloader = DataLoader(
                dataset,
                batch_size=config.val_batch_sizes[dataset_idx],
                pin_memory=False,
                shuffle=False,
            )
            # dataset = load_dataset(args.dataset_name_or_path, data_files='metadata.jsonl')['train']
            # for idx, sample in enumerate(dataset):
            for idx, batch in enumerate(dataloader):
                cur_idx = idx * config.val_batch_sizes[dataset_idx]
                if args.start_index and args.start_index > cur_idx:
                    if idx % 10 == 0:
                        print("skip", idx)
                    continue
                if args.test_cnt and args.test_cnt < cur_idx:
                    break
                print("###{}/{}/{}".format(dataset_name, dataset_idx, len(dataset_path_list)),
                      "{}/{}".format(idx, len(dataloader)))
                file_list = batch["file_name"]
                input_tensors = []
                gt_list = []
                image_sizes = []
                for fidx, file_name in enumerate(file_list):
                    im = Image.open(os.path.join(dataset_name_or_path, "validation", file_name))
                    image_sizes.append(im.size)
                    input_tensor = model.encoder.prepare_input(im, random_padding=False)
                    input_tensors.append(input_tensor)
                    gt_list.append(
                        T.postprocess_html_tag(json.loads(batch["ground_truth"][fidx])["gt_parse"]["text_sequence"]))
                input_tensors = torch.stack(input_tensors, dim=0)
                preds = model.inference(image_tensors=input_tensors, prompt=f"<s_tableocr>")["predictions"]
                print("len", preds)
                pred_list = [T.postprocess_html_tag(re.sub(r"(?:(?<=>) | (?=</s_))", "", pred['text_sequence'])) for
                             pred in preds]

                print(pred_list)
                print(gt_list)
                teds_all_list = teds_metric.batch(pred_list, gt_list)
                teds_struct_list = teds_metric_struct.batch(pred_list, gt_list)
                total_teds_all += teds_all_list
                total_teds_struct += teds_struct_list
                if len(dataset_path_list) > 1:
                    dataset_teds_all[dataset_name] += teds_all_list
                    dataset_teds_struct[dataset_name] += teds_struct_list

                for j, gt in enumerate(gt_list):
                    teds_all = teds_all_list[j]
                    teds_struct = teds_struct_list[j]
                    pred = pred_list[j]

                    item = {
                        "file_name": file_list[j],
                        "gt": gt,
                        "pred": pred,
                        "teds_all": teds_all,
                        "teds_struct": teds_struct,
                        "image_width": image_sizes[j][0],
                        "image_height": image_sizes[j][1],
                        "dataset_name": dataset_name
                    }
                    try:
                        output.write("{}\n".format(json.dumps(item)))
                    except:
                        item['index'] = idx
                        error_data.append(item)
                        traceback.print_exc()

                    if args.verbose:
                        print("")
                        print("#####", file_list[j])
                        print("===== gt")
                        print(gt)
                        print("===== pred")
                        print(pred)
                        print("===== gt structure")
                        print("".join(tag_re.findall(gt)))
                        print("===== pred structure")
                        print("".join(tag_re.findall(pred)))
                        print("===== teds all", teds_all)
                        print("===== teds structure", teds_struct)

    total_teds_mean = {
        "teds_all": np.mean(total_teds_all),
        "teds_structure": np.mean(total_teds_struct),
    }
    if len(dataset_path_list) > 1:
        for dataset_name in dataset_teds_all:
            total_teds_mean["teds_all_{}".format(dataset_name)] = np.mean(dataset_teds_all[dataset_name])
        for dataset_name in dataset_teds_struct:
            total_teds_mean["teds_structure_{}".format(dataset_name)] = np.mean(dataset_teds_struct[dataset_name])

    if error_data:
        pickle.dump(error_data, open(os.path.join(args.output_dir, "errors.pickle"), "wb+"))
        print("errors", len(error_data))

    print(total_teds_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--test_cnt", type=int, default=None)
    # 8666/9115
    args, left_argv = parser.parse_known_args()

    print("initializing config")
    config = Config(args.config)
    config.argv_update(left_argv)

    test(args, config)
