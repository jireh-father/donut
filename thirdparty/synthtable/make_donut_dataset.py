"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import glob
import shutil
import json
import os
from utils import html_util
from template import format_metadata
import argparse


def filter_html(html, remove_tag_in_content, remove_thead_tbody, remove_close_tag):
    if remove_tag_in_content:
        html, _ = html_util.remove_tag_in_table_cell(html)
    if remove_thead_tbody:
        html = html_util.remove_thead_tbody_tag(html)
    if remove_close_tag:
        html = html_util.remove_close_tags(html)
    html = html_util.remove_new_line_and_multiple_spaces(html)
    return html


def make_donut_dataset(dataset_root, output_dir, remove_tag_in_content, remove_thead_tbody, remove_close_tag):
    html_files = glob.glob(os.path.join(dataset_root, "*.html"))
    os.makedirs(output_dir, exist_ok=True)
    metadata_filepath = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_filepath, "w+", encoding='utf-8') as output_fp:
        for i, html_file in enumerate(html_files):
            print(i, len(html_file), html_file)
            with open(html_file, "r", encoding='utf-8') as html_fp:
                html = "\n".join(html_fp.readlines())
            html = filter_html(html, remove_tag_in_content, remove_thead_tbody, remove_close_tag)
            image_file = html_file.replace(".html", ".jpg")
            image_filename = os.path.basename(image_file)
            if not os.path.isfile(os.path.join(output_dir, os.path.basename(image_file))):
                if args.move_image_file:
                    shutil.move(image_file, output_dir)
                else:
                    shutil.copy(image_file, output_dir)
            metadata = format_metadata(image_filename=image_filename, keys=["text_sequence"], values=[html])
            if i > 0:
                output_fp.write("\n")
            json.dump(metadata, output_fp, ensure_ascii=False)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_roots', type=str,
                        default='D:\dataset\\table_ocr\crawling_filtered_val\google\en,D:\dataset\\table_ocr\crawling_filtered_val\google\kr,D:\dataset\\table_ocr\crawling_filtered_val\wiki\en,D:\dataset\\table_ocr\crawling_filtered_val\wiki\kr,D:\dataset\\table_ocr\crawling_filtered_val\danawa\kr')
    parser.add_argument('--output_dirs', type=str,
                        default='D:\dataset\\table_ocr\crawling_val\google_en_ori,D:\dataset\\table_ocr\crawling_val\google_kr_ori,D:\dataset\\table_ocr\crawling_val\wiki_en_ori,D:\dataset\\table_ocr\crawling_val\wiki_kr_ori,D:\dataset\\table_ocr\crawling_val\danawa_kr_ori')
    parser.add_argument('--remove_tag_in_content', action='store_true', default=True)
    parser.add_argument('--remove_thead_tbody', action='store_true', default=True)
    parser.add_argument('--remove_close_tag', action='store_true', default=True)
    parser.add_argument('--move_image_file', action='store_true', default=False)
    args = parser.parse_args()
    dataset_roots = args.dataset_roots.split(",")
    output_dirs = args.output_dirs.split(",")
    for dataset_root, output_dir in zip(dataset_roots, output_dirs):
        make_donut_dataset(dataset_root, output_dir, args.remove_tag_in_content, args.remove_thead_tbody,
                           args.remove_close_tag)
