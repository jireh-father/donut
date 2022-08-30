# https://huggingface.co/course/chapter6/2?fw=pt
from transformers import AutoTokenizer, XLMRobertaTokenizer, MBartTokenizer
import argparse
import json


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def convert_ptn_item_to_simple_html(item):
    table_tag = []

    text_set = set()
    total_texts = []
    i = 0
    tags = []
    max_col_span = 0
    max_row_span = 0
    while i < len(item['html']['structure']['tokens']):
        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag in ["<thead>", "</thead>", "<tbody>", "</tbody>"] or tag.startswith("</t"):
            continue
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
        else:
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
    return table_tag, total_texts, text_set, max_row_span, max_col_span


def main(args):
    if args.use_fast_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.token_path)
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.token_path)
    print(tokenizer)
    max_len = 0
    max_tag = None
    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        item = json.loads(line)
        table_tag, tmp_total_texts, text_set, tmp_max_row_span, tmp_max_col_span = convert_ptn_item_to_simple_html(
            item)
        cur_len = len(tokenizer.encode("".join(table_tag)))
        if cur_len > max_len:
            max_len = cur_len
            max_tag = table_tag
    print("max_len", max_len)
    print(max_tag)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--token_path', type=str, default="D:\dataset/table_ocr/pubtabnet/tokenizer_vocab_10k")
    parser.add_argument('--use_fast_tokenizer', action='store_true', default=True)

    main(parser.parse_args())
