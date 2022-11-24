import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json
import cv2
from bs4 import BeautifulSoup
import glob


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
    table_tag = []

    text_set = set()
    total_texts = []
    i = 0
    tags = []
    while i < len(item['html']['structure']['tokens']):

        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag == "<td":
            tag += item['html']['structure']['tokens'][i] + item['html']['structure']['tokens'][i + 1]
            i += 2
            tags.append(tag.strip())
        else:
            tags.append(tag)

    i = 0
    for tag in tags:
        table_tag.append(tag)
        if tag.startswith("<td"):
            text = remove_html_tags("".join(item['html']['cells'][i]['tokens'])).strip()
            if text:
                text = "".join(item['html']['cells'][i]['tokens']).strip()
                table_tag.append(text)
                text_set.update(set(text))
                total_texts.append(text)
            i += 1

    return table_tag, total_texts, text_set


def main(args):
    dataset_root = args.dataset_root
    output_dir = args.output_dir


    site_dirs = glob.glob(os.path.join(dataset_root, "*"))
    for site_dir in site_dirs:
        for lang_dir in glob.glob(os.path.join(site_dir, "*")):
            max_row_span = 0
            max_col_span = 0
            max_rows = 0
            max_cols = 0
            tmp_output_dir = os.path.join(output_dir, "{}_{}".format(os.path.basename(site_dir), os.path.basename(lang_dir)))
            os.makedirs(tmp_output_dir, exist_ok=True)
            for html_path in glob.glob(os.path.join(lang_dir, "*.html")):
                image_path = html_path.replace(".html", ".jpg")
                im = cv2.imread(image_path)
                height, width = im.shape[:2]

                html = "\n".join(open(html_path, "r", encoding='utf-8').readlines())
                bs = BeautifulSoup(html, 'html.parser')
                trs = bs.find_all("tr")
                nums_row = len(trs)
                nums_col = 0
                for td in trs[0].find_all("td"):
                    if td.has_attr("colspan"):
                        nums_col += int(td["colspan"])
                    else:
                        nums_col += 1

                td_rowspans = bs.select('td[rowspan]')
                tmp_max_row_span = 0
                for td_rowspan in td_rowspans:
                    if tmp_max_row_span < int(td_rowspan['rowspan']):
                        tmp_max_row_span = int(td_rowspan['rowspan'])

                td_colspans = bs.select('td[colspan]')
                tmp_max_col_span = 0
                for td_colspan in td_colspans:
                    if tmp_max_col_span < int(td_colspan['colspan']):
                        tmp_max_col_span = int(td_colspan['colspan'])

                if max_row_span < tmp_max_row_span:
                    max_row_span = tmp_max_row_span
                if max_col_span < tmp_max_col_span:
                    max_col_span = tmp_max_col_span

                if max_rows < nums_row:
                    max_rows = nums_row

                if max_cols < nums_col:
                    max_cols = nums_col

                result_item = {
                    'nums_col': nums_col,
                    'nums_row': nums_row,
                    'html': html,
                    'width': width,
                    'height': height,
                    "has_span": tmp_max_row_span > 1 or tmp_max_col_span > 1,
                    "has_row_span": tmp_max_row_span > 1,
                    "has_col_span": tmp_max_col_span > 1,
                    "max_col_span": tmp_max_col_span,
                    "max_row_span": tmp_max_row_span
                }

                json.dump(result_item, open(
                    os.path.join(tmp_output_dir, os.path.basename(html_path.replace(".html", ".json"))), "w+"))
            print("max_row_span", max_row_span)
            print("max_col_span", max_col_span)
            print("max_rows", max_rows)
            print("max_cols", max_cols)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="D:\dataset\\table_ocr\crawling_filtered_val")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\crawling_val_synthtiger_html_json")

    main(parser.parse_args())
