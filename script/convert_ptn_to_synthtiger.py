import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json
import cv2


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
    max_col_span = 0
    max_row_span = 0
    nums_row = 0
    nums_col = 0
    while i < len(item['html']['structure']['tokens']):

        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag == "<tr>":
            nums_row += 1
        # if use_thead:
        #     if tag.startswith("</t"):
        #         continue
        # else:
        #     if tag in ["<thead>", "</thead>", "<tbody>", "</tbody>"] or tag.startswith("</t"):
        #         continue
        if tag == "<td":
            split_tag = item['html']['structure']['tokens'][i].strip().split('"')
            num_spans = int(split_tag[1])
            if "col" in split_tag[0]:
                if num_spans > max_col_span:
                    max_col_span = num_spans
            else:
                if num_spans > max_row_span:
                    max_row_span = num_spans

            tag += item['html']['structure']['tokens'][i].strip() + item['html']['structure']['tokens'][i + 1]
            i += 2
            tags.append(tag.strip())
            nums_col += num_spans
        else:
            if tag == "<td>":
                nums_col += 1
            tags.append(tag)

    i = 0
    for tag in tags:
        table_tag.append(tag)
        if tag.startswith("<td"):
            text = remove_html_tags("".join(item['html']['cells'][i]['tokens'])).strip()
            if text:
                table_tag.append(text)
                text_set.update(set(text))
                total_texts.append(text)
            i += 1
    return table_tag, total_texts, text_set, max_row_span, max_col_span, nums_row, int(nums_col / nums_row)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    max_row_span = 0
    max_col_span = 0
    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        if args.test_cnt and i >= args.test_cnt:
            break
        item = json.loads(line)
        table_tag, tmp_total_texts, text_set, tmp_max_row_span, tmp_max_col_span, nums_row, nums_col = convert_ptn_item_to_simple_html(
            item)
        file_name = item['filename']
        if item['split'] == "train":
            image_path = os.path.join(args.image_dir, "train", file_name)
        else:
            image_path = os.path.join(args.image_dir, "validation", file_name)
        im = cv2.imread(image_path)
        height, width = im[:2]

        if max_row_span < tmp_max_row_span:
            max_row_span = tmp_max_row_span
        if max_col_span < tmp_max_col_span:
            max_col_span = tmp_max_col_span
        result_item = {
            'nums_col': nums_col,
            'nums_row': nums_row,
            'html': "".join(table_tag),
            'width': width,
            'height': height,
            "has_span": tmp_max_row_span > 1 or tmp_max_col_span > 1,
            "has_row_span": tmp_max_row_span > 1,
            "has_col_span": tmp_max_col_span > 1,
            "max_col_span": tmp_max_col_span,
            "max_row_span": tmp_max_row_span
        }

        json.dump(result_item, open(os.path.join(args.output_dir, os.path.splitext(file_name)[0] + ".json"), "w+"))

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
