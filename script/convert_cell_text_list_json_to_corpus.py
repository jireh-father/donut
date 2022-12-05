import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json
import re
import glob
import sys

sys.path.append('../thirdparty/synthtable')
from utils import html_util

empty_char_regex = re.compile(r'[\u0000-\u001F\u007F]')
korean_regex = re.compile(r'[\uAC00-\uD7A3\u1100-\u11FF\u3131-\u318F]')
japanese_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]')
chinese_regex = re.compile(
    r'[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900-\uFAFF\U00020000-\U0002A6DF\U0002F800-\U0002FA1F]')
english_regex = re.compile(
    r'[\u0000-\u007E\u00A1-\u00BF\u2160-\u216B\u2170-\u217B\u2190-\u2199\u2200-\u22FF\u2460-\u2473\u24B6-\u24E9\u00D7\u00F7\u203B\u2022]')
latin_number_regex = re.compile(
    r'[\u0030-\u0039\u0041-\u005A\u0061-\u007A]')
except_korean_english_regex = re.compile(
    r'[^\u0000-\u007E\u00A1-\u00BF\u2160-\u216B\u2170-\u217B\u2190-\u2199\u2200-\u22FF\u2460-\u2473\u24B6-\u24E9\uAC00-\uD7A3\u1100-\u11FF\u3131-\u318F\u0030-\u0039\u0041-\u005A\u0061-\u007A\u00D7\u00F7\u203B\u2022]')
multiple_space_regex = re.compile(' +')


def _pad_text_side_with_spaces(tags):
    for tag in tags:
        if tag.name:
            if hasattr(tag, "contents"):
                _pad_text_side_with_spaces(tag.contents)
        else:
            tag.replace_with(" {} ".format(tag.text))


def main(args):
    text_list = json.load(open(args.json_file, encoding='utf-8'))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w+", encoding='utf-8') as fp:
        for i, text in enumerate(text_list):
            if re.search(except_korean_english_regex, text):
                print("except chars", text)
                continue
            if i > 0:
                fp.write("\n")
            fp.write(text)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        default="D:\dataset\\table_ocr\gmarket_key_corpus/machine_spec_keys.json")
    parser.add_argument('--output_path', type=str,
                        default="D:\dataset\\table_ocr\gmarket_key_corpus/gmarket_table_key_corpus.txt")
    main(parser.parse_args())
