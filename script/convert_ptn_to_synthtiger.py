import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json
import cv2
from bs4 import BeautifulSoup


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
    max_row_span = 0
    max_col_span = 0
    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        if args.test_cnt and i >= args.test_cnt:
            break
        item = json.loads(line)
        table_tag, tmp_total_texts, text_set = convert_ptn_item_to_simple_html(item)
        file_name = item['filename']
        if item['split'] == "train":
            image_path = os.path.join(args.image_dir, "train", file_name)
        else:
            image_path = os.path.join(args.image_dir, "validation", file_name)
        im = cv2.imread(image_path)
        height, width = im.shape[:2]

        html = "<table>{}</table>".format("".join(table_tag))
        bs = BeautifulSoup(html, 'html.parser')
        trs = bs.find_all("tr")
        nums_row = len(trs)
        max_col = 0
        for tr in trs:
            for tds in tr.find_all("td"):
                if max_col < len(tds):
                    max_col = len(tds)
        nums_col = max_col

        td_rowspans = bs.select('td[rowspan]')
        tmp_max_row_span = 0
        for td_rowspan in td_rowspans:
            if tmp_max_row_span < td_rowspan['rowspan']:
                tmp_max_row_span = td_rowspan['rowspan']

        td_colspans = bs.select('td[colspan]')
        tmp_max_col_span = 0
        for td_colspan in td_colspans:
            if tmp_max_col_span < td_colspan['colspan']:
                tmp_max_col_span = td_colspan['colspan']

        if max_row_span < tmp_max_row_span:
            max_row_span = tmp_max_row_span
        if max_col_span < tmp_max_col_span:
            max_col_span = tmp_max_col_span

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
        output_dir = os.path.join(args.output_dir, item['split'])
        os.makedirs(output_dir, exist_ok=True)
        json.dump(result_item, open(os.path.join(output_dir, os.path.splitext(file_name)[0] + ".json"), "w+"))

    print("max_row_span", max_row_span)
    print("max_col_span", max_col_span)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--image_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")

    parser.add_argument('--test_cnt', type=int, default=None)
    main(parser.parse_args())
