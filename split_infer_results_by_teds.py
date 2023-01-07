"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from transformers import AutoTokenizer, XLMRobertaTokenizer
import json
import pandas as pd
import imgkit
import argparse
import shutil
import os

html_template = """
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body>
            {}
        </body>
        </html>
        """

def test(args):
    data_file = open(args.result_file, encoding="utf-8")

    items = []
    for i, line in enumerate(data_file):
        item = json.loads(line)
        items.append(item)
    print(len(items))

    df = pd.DataFrame.from_records(items)

    for index, row in df.iterrows():
        if index % 100 == 0:
            print(index, len(df))
        dataset_name = row['dataset_name']
        file_name = row['file_name']
        only_file_name = os.path.splitext(file_name)[0]
        pred_html = row['pred']
        teds_score = "10" if row['teds_all'] >= 1 else str(row['teds_all'])[2:3] + "0"

        cur_output_dir = os.path.join(args.output_dir, "vis_dataset_and_teds", dataset_name, teds_score)
        os.makedirs(cur_output_dir, exist_ok=True)
        image_path = os.path.join(args.dataset_root, dataset_name, "validation", file_name)
        shutil.copy(image_path, cur_output_dir)

        output_html = html_template.format(pred_html.replace("<table>", '<table border="1">'))
        html_path = os.path.join(cur_output_dir, only_file_name + ".html")
        with open(html_path, "w+", encoding='utf-8') as fp:
            fp.write(output_html)
        html_jpg_path = os.path.join(cur_output_dir, only_file_name + "_html.jpg")
        imgkit.from_file(html_path, html_jpg_path, options={"xvfb": ""})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=True)
    parser.add_argument("--dataset_root", type=str, default=None)

    test(parser.parse_args())
