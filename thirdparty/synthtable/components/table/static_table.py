import os.path
import json
import traceback
from bs4 import BeautifulSoup

from PIL import Image
from utils import image_util
from synthtiger.components.component import Component

from utils.path_selector import PathSelector


class StaticTable(Component):
    def __init__(self, config_selectors, config):

        super(StaticTable, self).__init__()
        self.html_path_selector = PathSelector(config_selectors['html']['paths'].values,
                                               config_selectors['html']['weights'].values, exts=['.json'])
        self.image_path_selector = PathSelector(config_selectors['image']['paths'].values, None,
                                                exts=['.jpg', '.png'])
        self.html_path_shuffle = config["html"]["shuffle"] if "shuffle" in config["html"] else True
        self.html_file_idx = 0
        self.min_image_size_ratio = config_selectors['image']['min_image_size_ratio'].values

        self.config_selectors = config_selectors

        self.min_rows = config_selectors['html']['min_row'].select()
        self.max_rows = config_selectors['html']['max_row'].select()
        self.min_cols = config_selectors['html']['min_col'].select()
        self.max_cols = config_selectors['html']['max_row'].select()
        self.max_col_span = config_selectors['html']['max_col_span'].select() if 'max_col_span' in config_selectors[
            'html'] else None
        self.max_row_span = config_selectors['html']['max_row_span'].select() if 'max_row_span' in config_selectors[
            'html'] else None
        self.max_empty_cell_ratio = config_selectors['html']['max_empty_cell_ratio'].select()
        self.max_image_width = config_selectors['html']['max_image_width'].select()
        self.max_image_height = config_selectors['html']['max_image_height'].select()

        self.has_span = self.config_selectors['html']['has_span']
        self.has_col_span = self.config_selectors['html']['has_col_span']
        self.has_row_span = self.config_selectors['html']['has_row_span']
        self.config = config

    def _get_image_path(self, html_path, key):
        html_file_name = os.path.splitext(os.path.basename(html_path))[0]
        image_path = self.image_path_selector.get_path(key)
        for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            path = os.path.join(image_path, html_file_name) + "." + ext
            if os.path.isfile(path):
                return path
        return None

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if self.html_path_shuffle:
            html_path, key, idx = self.html_path_selector.select()
        else:
            html_path = self.html_path_selector.get(0, self.html_file_idx)
            key = 0
            if len(self.html_path_selector.paths[key]) <= self.html_file_idx + 1:
                self.html_file_idx = 0
            else:
                self.html_file_idx += 1
        image_path = self._get_image_path(html_path, key)
        if not image_path:
            return self.sample(meta)

        meta["html_path"] = html_path
        meta["image_path"] = image_path
        meta["min_image_size_ratio"] = self.min_image_size_ratio[key]
        meta['has_span'] = self.has_span.on()
        meta['has_row_span'] = self.has_row_span.on()
        meta['has_col_span'] = self.has_col_span.on()
        meta['max_image_height'] = self.max_image_height
        meta['max_image_width'] = self.max_image_width

        return meta

    def apply(self, layers, meta=None):
        target_size = meta['size']
        while True:
            # try:
            meta = self.sample(meta)
            # except Exception as e:
            #     traceback.print_exc()
            #     continue

            html_json_path = meta['html_path']
            html_json = json.load(open(html_json_path), encoding='utf-8')

            if self.max_empty_cell_ratio:
                bs = BeautifulSoup(html_json['html'], 'html.parser')
                num_empty_tds = 0
                tds = bs.find_all("td")
                for td in tds:
                    if not td.text.strip():
                        num_empty_tds += 1
                if num_empty_tds / len(tds) > self.max_empty_cell_ratio:
                    continue

            if self.html_path_shuffle:
                if self.min_cols > html_json['nums_col'] or self.max_cols < html_json['nums_col']:
                    continue
                if self.min_rows > html_json['nums_row'] or self.max_rows < html_json['nums_row']:
                    continue

                if self.max_col_span and html_json['max_col_span'] > self.max_col_span:
                    continue

                if self.max_row_span and html_json['max_row_span'] > self.max_row_span:
                    continue

                if meta['has_span'] != html_json['has_span']:
                    continue

                width_limit = html_json['width'] * meta["min_image_size_ratio"]
                height_limit = html_json['height'] * meta["min_image_size_ratio"]

                target_width, target_height = target_size

                if width_limit > target_width or height_limit > target_height:
                    continue

                if meta['has_span']:
                    if meta['has_row_span'] and not html_json['has_row_span']:
                        continue
                    if meta['has_col_span'] and not html_json['has_col_span']:
                        continue

            html = html_json['html']
            meta['html'] = html
            meta['has_span'] = html_json['has_span']
            meta['nums_col'] = html_json['nums_col']
            meta['nums_row'] = html_json['nums_row']
            meta['width'] = html_json['width']
            meta['height'] = html_json['height']
            meta['has_row_span'] = html_json['has_row_span']
            meta['has_col_span'] = html_json['has_col_span']
            image_path = meta['image_path']
            break

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image, target_size = image_util.resize_keeping_aspect_ratio(image, target_size, Image.ANTIALIAS)
        meta["effect_config"] = self.config["effect"]
        for layer in layers:
            layer.render_table(image=image, meta=meta)
