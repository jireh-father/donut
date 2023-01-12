"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import glob
import json
import synthtiger
import os
from typing import Any, List

import numpy as np
from elements import Background, Document
from PIL import Image
from synthtiger import components, layers, templates
from utils import html_util
from selenium import webdriver


def format_metadata(image_filename: str, keys: List[str], values: List[Any]):
    """
    Fit gt_parse contents to huggingface dataset's format
    keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
    Args:
        keys: List of task_name
        values: List of actual gt data corresponding to each task_name
    """
    assert len(keys) == len(values), "Length does not match: keys({}), values({})".format(len(keys), len(values))

    _gt_parse_v = dict()
    for k, v in zip(keys, values):
        _gt_parse_v[k] = v
    gt_parse = {"gt_parse": _gt_parse_v}
    gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
    metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
    return metadata

class SynthTable(templates.Template):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}

        self.html_output = config.get("html_output")
        self.quality = config.get("quality", [50, 95])
        self.save_meta = config.get("save_meta", False)
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )
        self.table_html_synth = False
        self.selenium_driver = None
        # if config["document"]["content"]["table"]["synth"]["weight"] > 0:
        #     self.table_html_synth = True
        #     options = webdriver.ChromeOptions()
        #     options.add_argument('--headless')
        #     options.add_argument('--no-sandbox')
        #     options.add_argument('--disable-dev-shm-usage')
        #     options.add_argument('--detach_driver')
        #     self.selenium_driver = webdriver.Chrome('chromedriver', options=options)
        #     self.selenium_driver.implicitly_wait(0.5)

    def __del__(self):
        if self.table_html_synth:
            self.selenium_driver.close()
            self.selenium_driver.quit()

    def _filter_html(self, html, bs=None):
        if self.html_output["remove_tag_in_content"]:
            html, _ = html_util.remove_tag_in_table_cell(html, bs)

        if self.html_output["remove_thead_tbody"]:
            html = html_util.remove_thead_tbody_tag(html)

        if self.html_output["remove_close_tag"]:
            html = html_util.remove_close_tags(html)

        html = html_util.remove_new_line_and_multiple_spaces(html)

        return html

    def generate(self):
        # todo:
        '''
        pubtabnet 변형 데이터 생성
        1. wild
        2. wild hard

        수집한 table html & table image 어떻게 전처리할지 고민

        ## later to do
        아이콘이나 이미지 넣기<i> 태그

        change html structure. span
        '''

        table_layer, bg_size = self.document.generate(self.selenium_driver)

        bg_layer, bg_image_meta, bg_effect_meta = self.background.generate(bg_size)
        table_layer.meta['bg_image'] = bg_image_meta
        table_layer.meta['bg_effect'] = bg_effect_meta

        table_html = self._filter_html(table_layer.html,
                                       table_layer.meta['html_bs'] if 'html_bs' in table_layer.meta else None)

        document_space = np.clip(bg_size - table_layer.size, 0, None)
        table_layer.left = np.random.randint(document_space[0] + 1)
        table_layer.top = np.random.randint(document_space[1] + 1)
        roi = np.array(table_layer.quad, dtype=int)

        layer = layers.Group([table_layer, bg_layer]).merge()
        effect_meta = self.effect.apply([layer])

        table_layer.meta['template_effect'] = effect_meta

        image = layer.output(bbox=[0, 0, *bg_size])
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        table_layer.meta['quality'] = quality

        if 'html_bs' in table_layer.meta:
            del table_layer.meta['html_bs']

        data = {
            "image": image,
            "label": table_html,
            "quality": quality,
            "roi": roi,
            "meta": table_layer.meta,
            "save_meta": self.save_meta
        }

        return data

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        roi = data["roi"]
        meta = data["meta"]
        save_meta = data['save_meta']

        # split
        # output_dirpath = os.path.join(root, "train")
        output_dirpath = root

        file_idx = idx

        # split_prob = np.random.rand()
        # for _idx, (split, ratio) in enumerate(zip(self.splits, self.split_ratio)):
        #     if split_prob < ratio:
        #         output_dirpath = os.path.join(root, split)
        #         file_idx = self.split_indexes[_idx]
        #         self.split_indexes[_idx] += 1
        #         break

        # save image
        image_filename = f"image_{file_idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save meta json file
        if save_meta:
            meta_filename = f"image_{file_idx}_meta.json"
            meta_filepath = os.path.join(output_dirpath, meta_filename)
            json.dump(meta, open(meta_filepath, "w+", encoding='utf-8'))

        # save metadata (gt_json)
        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        metadata = format_metadata(image_filename=image_filename, keys=["text_sequence"], values=[label])
        with open(metadata_filepath, "a", encoding='utf-8') as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def load(self, data):
        image = data["image"]
        label = data["label"]
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        return image, label

    def end_save(self, root):
        pass

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--model_path', type=str, default=None)
    # parser.add_argument('--csv_file', type=str, default=None)
    # parser.add_argument('--model_name', type=str, default='efficientnet-b2')
    # parser.add_argument('--input_size', type=int, default=260)
    # parser.add_argument('--line_width', type=int, default=1)
    # parser.add_argument('--use_cuda', action='store_true', default=False)

    synth_table = SynthTable(synthtiger.read_config("config_danawa_kr_synth_val_remove_img_tag_easy_test_on_pc_nopoly.yaml"))

    for i in range(100):
        print(i)
        try:
            synth_table.save("./output_nopoly", synth_table.generate(), i)
        except Exception as e:
            continue
