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

def _pad_text_side_with_spaces(tags):
    for tag in tags:
        if tag.name:
            if hasattr(tag, "contents"):
                _pad_text_side_with_spaces(tag.contents)
        else:
            tag.replace_with(" {} ".format(tag.text))


def main(args):
    json_dirs = args.json_dirs.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    for json_dir in json_dirs:
        dataset_dir = os.path.basename(json_dir)
        output_path = os.path.join(args.output_dir, dataset_dir + "_tokenizer_corpus.txt")
        with open(output_path, "w+", encoding="utf-8") as f:
            json_files = glob.glob(os.path.join(json_dir, "*.json"))
            cell_text_set = set()
            for i, json_file in enumerate(json_files):
                print(i, len(json_files), json_file)
                data = json.load(open(json_file, "r", encoding="utf-8"))
                html = data['html']
                _, bs = html_util.remove_tag_in_table_cell(html)

                _pad_text_side_with_spaces(bs.contents)

                tds = bs.find_all("td")
                for td in tds:
                    cell_text = "".join("<img>" if v.name == 'img' else v.text for v in td.contents)
                    cell_text = re.sub(empty_char_regex, "", cell_text)
                    cell_text = html_util.remove_multiple_spaces(cell_text).strip()
                    if not cell_text:
                        continue
                    cell_text_set.add(cell_text)

                text = bs.text
                text = re.sub(html_util.new_line_regex, " ", text)
                text = re.sub(empty_char_regex, "", text)
                text = html_util.remove_multiple_spaces(text).strip()
                print(text)
                if i > 0:
                    f.write("\n")
                f.write(text)
            cell_text_list = list(cell_text_set)
            cell_text_list.sort()
            with open(os.path.join(args.output_dir, dataset_dir + "_cell_text_corpus.txt"), "w+",
                      encoding="utf-8") as cf:
                for j, cell_text in enumerate(cell_text_list):
                    if j > 0:
                        cf.write("\n")
                    cf.write(cell_text)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dirs', type=str,
                        default="D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/auction_kr,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/danawa_kr,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/gmarket_kr,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/google_en,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/google_kr,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/wiki_en,D:\dataset\\table_ocr\crawling_train_synthtiger_html_json/wiki_kr")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\crawling_train_corpus")

    main(parser.parse_args())
