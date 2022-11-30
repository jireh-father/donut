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
    charset_unicodes = args.charset_unicodes.split(",")
    charset = []
    for unicode in charset_unicodes:
        if len(unicode) == 4:
            charset.append(chr(int(unicode, 16)))
        else:
            unicode_rage = unicode.split("-")
            for i in range(int(unicode_rage[0], 16), int(unicode_rage[1], 16) + 1):
                charset.append(chr(i))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fp = open(args.output_path, "w+", encoding='utf-8')
    for c in charset:
        fp.write(c)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str,
                        default="../thirdparty/synthtable/resources/charset/alphanum_special_korean.txt")

    parser.add_argument('--charset_unicodes', type=str,
                        default='0020-007E,00A1-00AC,00AE-00BF,2160-216B,2170-217B,2190-2199,2200-22FF,AC00-D7A3,3131-3163,00D7,00F7,203B,2022')

    main(parser.parse_args())
