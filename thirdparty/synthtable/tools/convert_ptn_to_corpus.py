import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# tokenizers
# https://wikidocs.net/166826
# https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer
# https://misconstructed.tistory.com/80

def convert_ptn_item_to_simple_html(item):
    thead_text_set = set()
    tbody_text_set = set()

    i = 0
    tags = []

    while i < len(item['html']['structure']['tokens']):
        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag.startswith("</t"):
            continue
        if tag == "<td":
            tag += item['html']['structure']['tokens'][i] + item['html']['structure']['tokens'][i + 1]
            # tag += item['html']['structure']['tokens'][i].strip() + item['html']['structure']['tokens'][i + 1]
            i += 2
            tags.append(tag.strip())
        else:
            tags.append(tag)

    i = 0
    is_thead = True
    max_text_length = 0
    for tag in tags:
        if tag == "<tbody>":
            is_thead = False
        if tag.startswith("<td"):
            text = remove_html_tags("".join(item['html']['cells'][i]['tokens'])).strip()

            if text:
                if len(text) > max_text_length:
                    max_text_length = len(text)
                if is_thead:
                    thead_text_set.add(text)
                else:
                    tbody_text_set.add(text)
            i += 1
    return thead_text_set, tbody_text_set, max_text_length


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_thead_corpus_output = os.path.join(args.output_dir, "train_thead_corpus.txt")
    val_thead_corpus_output = os.path.join(args.output_dir, "val_thead_corpus.txt")
    train_tbody_corpus_output = os.path.join(args.output_dir, "train_tbody_corpus.txt")
    val_tbody_corpus_output = os.path.join(args.output_dir, "val_tbody_corpus.txt")

    train_thead_text_list = set()
    train_tbody_text_list = set()
    val_thead_text_list = set()
    val_tbody_text_list = set()
    max_text_length = 0
    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        if args.test_cnt and i >= args.test_cnt:
            break
        item = json.loads(line)

        thead_text_set, tbody_text_set, tmp_max_text_length = convert_ptn_item_to_simple_html(item)
        if max_text_length < tmp_max_text_length:
            max_text_length = tmp_max_text_length
        if item['split'] == "train":
            train_thead_text_list.update(thead_text_set)
            train_tbody_text_list.update(tbody_text_set)
        else:
            val_thead_text_list.update(thead_text_set)
            val_tbody_text_list.update(tbody_text_set)
    train_thead_text_list = list(train_thead_text_list)
    train_thead_text_list.sort()

    train_tbody_text_list = list(train_tbody_text_list)
    train_tbody_text_list.sort()

    val_thead_text_list = list(val_thead_text_list)
    val_thead_text_list.sort()

    val_tbody_text_list = list(val_tbody_text_list)
    val_tbody_text_list.sort()
    open(train_thead_corpus_output, "w+", encoding='utf-8').writelines(
        ["\n" + text for i, text in enumerate(train_thead_text_list) if i > 0])
    open(train_tbody_corpus_output, "w+", encoding='utf-8').writelines(
        ["\n" + text for i, text in enumerate(train_tbody_text_list) if i > 0])
    open(val_thead_corpus_output, "w+", encoding='utf-8').writelines(
        ["\n" + text for i, text in enumerate(val_thead_text_list) if i > 0])
    open(val_tbody_corpus_output, "w+", encoding='utf-8').writelines(
        ["\n" + text for i, text in enumerate(val_tbody_text_list) if i > 0])
    print("max_text_length", max_text_length)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")

    parser.add_argument('--test_cnt', type=int, default=None)
    main(parser.parse_args())
