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
    json_dirs = args.json_dirs.split(",")

    dataset_names = args.dataset_names.split(",") if args.dataset_names else None
    os.makedirs(args.output_dir, exist_ok=True)
    for dir_idx, json_dir in enumerate(json_dirs):
        print(json_dir)
        dataset_dir = os.path.basename(json_dir)
        output_path = os.path.join(args.output_dir, (dataset_names[dir_idx] if dataset_names else dataset_dir) + "_tokenizer_corpus.txt")
        if args.save_tokenizer_corpus:
            tokenizer_fp = open(output_path, "w+", encoding="utf-8")
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        cell_text_set = set()
        for i, json_file in enumerate(json_files):
            # if i % 100 == 0:
            #     print(i, len(json_files), json_file)
            data = json.load(open(json_file, "r", encoding="utf-8"))
            html = data['html']
            if args.save_cell_text_corpus:
                _, bs = html_util.remove_tag_in_table_cell(html, remove_img_tag=True)

                _pad_text_side_with_spaces(bs.contents)

                tds = bs.find_all("td")
                tmp_cell_text_set = set()
                skip_cell_text = False
                for td in tds:
                    cell_text = "".join(" " if v.name == 'img' else v.text for v in td.contents)
                    cell_text = re.sub(empty_char_regex, "", cell_text)
                    if re.search(except_korean_english_regex, cell_text):
                        print("except chars", cell_text)
                        skip_cell_text = True
                        break
                    cell_text = html_util.remove_multiple_spaces(cell_text).strip()

                    if not cell_text:
                        continue
                    tmp_cell_text_set.add(cell_text)
                if not skip_cell_text:
                    cell_text_set.update(tmp_cell_text_set)

            if args.save_tokenizer_corpus:
                _, bs = html_util.remove_tag_in_table_cell(html, remove_img_tag=False)

                _pad_text_side_with_spaces(bs.contents)
                text = bs.text
                text = re.sub(html_util.new_line_regex, " ", text)
                text = re.sub(empty_char_regex, "", text)
                if re.search(except_korean_english_regex, text):
                    print("except text", text)
                    continue
                text = html_util.remove_multiple_spaces(text).strip()
                # print(text)
                if i > 0:
                    tokenizer_fp.write("\n")
                tokenizer_fp.write(text)
        if args.save_tokenizer_corpus:
            tokenizer_fp.close()

        if args.save_cell_text_corpus:
            cell_text_list = list(cell_text_set)
            cell_text_list.sort()
            with open(os.path.join(args.output_dir,
                                   (dataset_names[dir_idx] if dataset_names else dataset_dir) + "_cell_text_corpus.txt"),
                      "w+",
                      encoding="utf-8") as cf:
                for j, cell_text in enumerate(cell_text_list):
                    if j > 0:
                        cf.write("\n")
                    cf.write(cell_text)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dirs', type=str,
                        default="D:\dataset\\table_ocr\crawling_synthtiger_html_json/auction_kr/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/danawa_kr/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/gmarket_kr/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/google_en/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/google_kr/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/wiki_en/train,D:\dataset\\table_ocr\crawling_synthtiger_html_json/wiki_kr/train")
    parser.add_argument('--dataset_names', type=str,
                        default="auction_kr,danawa_kr,gmarket_kr,google_en,google_kr,wiki_en,wiki_kr")
    parser.add_argument('--output_dir', type=str,
                        default="D:\dataset\\table_ocr\crawling_train_corpus_without_except_chars_and_img_tags")
    parser.add_argument('--save_tokenizer_corpus', action='store_true', default=False)
    parser.add_argument('--save_cell_text_corpus', action='store_true', default=True)
    main(parser.parse_args())
