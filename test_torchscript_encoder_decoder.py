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
import re
import teds as T

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
    decoder = ori_model.decoder
    input_ids = decoder.tokenizer("<s_tableocr>", add_special_tokens=False, return_tensors="pt")["input_ids"]
    # max_len = 3000
    # input_ids = torch.tensor([[decoder.tokenizer.get_vocab()["<s_tableocr>"]] + [decoder.pad_token_id] * (max_len - 1)])

    vocab = ori_model.decoder.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    image_files = glob.glob(args.images_pattern)

    im = Image.open(image_files[0])
    if im.mode != "RGB":
        im = im.convert("RGB")
    input_tensors = []
    input_tensor = ori_model.encoder.prepare_input(im, random_padding=False)
    input_tensors.append(input_tensor)
    input_image = torch.stack(input_tensors, dim=0)

    device = 'cpu'

    input_ids = input_ids.to(device)
    input_image = input_image.to(device)

    encoder = torch.jit.load(args.encoder_model_path)
    decoder = torch.jit.load(args.decoder_model_path)
    start = time.time()
    hidden_state = encoder(input_image)
    print(time.time() - start)
    print("hidden_state", hidden_state)
    print(hidden_state.shape)

    static_bad_words_mask = None
    total_time = time.time()
    cur_len = 1
    while True:
        start = time.time()
        logits = decoder(input_ids, hidden_state, torch.tensor([cur_len - 1]).to(device))
        print(time.time() - start)
        print("logits", logits)
        print(logits.shape)
        # next_token_logits = logits[:, cur_len-1, :]
        # next_token_logits = logits[:, -1, :]
        next_token_logits = logits
        print("next_token_logits", next_token_logits.shape)

        if static_bad_words_mask is None:
            static_bad_words_mask = torch.zeros(next_token_logits.shape[0])
            static_bad_words_mask[ori_model.decoder.unk_token_id] = 1
            static_bad_words_mask = static_bad_words_mask.to(next_token_logits.device).bool()

        next_token_logits = next_token_logits.masked_fill(static_bad_words_mask, -float("inf"))

        next_tokens = torch.argmax(next_token_logits, dim=-1)
        # input_ids[:, cur_len] = next_tokens[0]
        print("next_tokens", next_tokens.shape)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)[:, None]], dim=-1)
        cur_len = cur_len + 1

        if next_tokens == ori_model.decoder.eos_token_id:
            break
        if cur_len >= 3000:
            break
    print("total time", time.time() - total_time)
    preds = input_ids[:, :cur_len]
    for i, pred in enumerate(preds):
        print(pred)
        pred = [inv_vocab.get(int(p), "") for p in pred]
        print(pred)
        # pred = model.decoder.tokenizer.decode(pred)
        # if pred.startswith("<s_tableocr> "):
        #     pred = pred[len("<s_tableocr> "):]
        # if pred.endswith("</s>"):
        #     pred = pred[:-len("</s>")]
        pred = "".join(pred[1:-1]).replace("▁", " ")
        print(pred)
        pred = T.postprocess_html_tag(re.sub(r"(?:(?<=>) | (?=</s_))", "", pred))
        print(pred)
    #
    print(input_ids)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./config/train_swinv2_realworld_synth_for_test_in_pc.yaml")
    parser.add_argument("--images_pattern", type=str, default='D:\dataset\\table_ocr\\test_sample_en_one2/*')
    parser.add_argument('--encoder_model_path', default='D:\\result\\tableocr\\android/swin_encoder.ptl', type=str)
    parser.add_argument('--decoder_model_path', default='D:\\result\\tableocr\\android/only_decoder_no_attention_mask_with_return_index.ptl', type=str)

    args, left_argv = parser.parse_known_args()
    main(args, left_argv)
