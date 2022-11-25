import os
import math
import json
import traceback
import uuid
from synthtiger.components.component import Component
from utils.path_selector import PathSelector
from elements.paper import Paper
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
from utils import html_util
from synthtiger import components
import random
from utils.html_util import remove_tags
from utils.charset import Charset
from utils.html_util import convert_bs_to_html_string


class SynthTable(Component):
    def __init__(self, config_selectors, config):
        super().__init__()
        self.meta = None
        self.global_style = None
        self.content_text = None
        self.config = config
        self.config_selectors = config_selectors
        self.html_path_selector = PathSelector(config_selectors['html']['paths'].values,
                                               config_selectors['html']['weights'].values, exts=['.json'])

        self.html_path_shuffle = config["html"]["shuffle"] if "shuffle" in config["html"] else True
        self.html_file_idx = 0

        self.html_charset = None
        if 'charset' in config_selectors['html'] and config_selectors['html']['charset']:
            self.html_charset = Charset(config_selectors['html']['charset'])

        # styles
        for background_config in config_selectors['style']['global']['absolute']['table_wrapper']['background'].values:
            name = background_config['name']

            if name == 'paper':
                paper_config = config["style"]["global"]["absolute"]['table_wrapper']["background"]["paper"]
                self.paper = Paper({k: paper_config[k] for k in paper_config if k != "weight"})
            elif name == 'gradient':
                self.gradient_bg = background_config['config']

        # color set
        self.dark_colors = config_selectors['style']['color_set']['dark']
        self.light_colors = config_selectors['style']['color_set']['light']
        self.global_color_mode = config_selectors['style']['global']['absolute']['table']['color_mode']

        # global absolute thead
        self.absolute_style = defaultdict(dict)

        # common styles
        self.min_rows = config_selectors['html']['min_row'].select()
        self.max_rows = config_selectors['html']['max_row'].select()
        self.min_cols = config_selectors['html']['min_col'].select()
        self.max_cols = config_selectors['html']['max_row'].select()
        self.max_empty_cell_ratio = config_selectors['html']['max_empty_cell_ratio'].select()
        self.max_image_width = config_selectors['html']['max_image_width'].select()
        self.max_image_height = config_selectors['html']['max_image_height'].select()
        self.max_big_image_ratio = config_selectors['html']['max_big_image_ratio'].select()
        self.max_big_image_ratio_when_less_cells = config_selectors['html'][
            'max_big_image_ratio_when_less_cells'].select()
        self.num_less_cell = config_selectors['html']['num_less_cell'].select()

        self.has_span = config_selectors['html']['has_span']
        self.has_col_span = config_selectors['html']['has_col_span']
        self.has_row_span = config_selectors['html']['has_row_span']

        self.min_font_size = self.config['style']['global']['css']['table']['font-size']['values'][0]
        self.max_font_size = self.config['style']['global']['css']['table']['font-size']['values'][1]

        tmp_path = config_selectors['html']['tmp_path'].select()
        os.makedirs(tmp_path, exist_ok=True)

        self.font = components.BaseFont(**config["style"].get("font", {}))

        if config["html"]["structure"]["synth_structure"]["weight"] > 0 or config["html"]["synth_content"]["prob"] > 0:
            self.synth_structure_config = None
            corpus_dict = defaultdict(dict)
            for thead_or_tbody in ["thead", "tbody"]:
                for corpus_type in config["html"]["synth_content"]["corpus"][thead_or_tbody].keys():
                    corpus_config = config["html"]["synth_content"]["corpus"][thead_or_tbody][corpus_type]
                    if corpus_type == "length_augmentable":
                        corpus = components.LengthAugmentableCorpus(
                            **{k: corpus_config[k] for k in corpus_config if k != "weight"})
                    elif corpus_type == "char_augmentable":
                        corpus = self.thead_char_aug_corpus = components.CharAugmentableCorpus(
                            **{k: corpus_config[k] for k in corpus_config if k != "weight"})
                    elif corpus_type == "base":
                        corpus = components.BaseCorpus(
                            **{k: corpus_config[k] for k in corpus_config if k != "weight"})
                    corpus_dict[thead_or_tbody][corpus_type] = corpus
            self.corpus_dict = corpus_dict

            self.thead_corpus_selector = config_selectors['html']['synth_content'].get()['corpus']['thead']
            self.tbody_corpus_selector = config_selectors['html']['synth_content'].get()['corpus']['tbody']
            self.empty_cell_switch = config_selectors['html']['synth_content'].get()['empty_cell']
            self.thead_bold_switch = config_selectors['html']['synth_content'].get()['thead_bold']
            self.synth_cell_switch = config_selectors['html']['synth_content'].get()['synth_cell']
            self.shuffle_cells_switch = config_selectors['html']['synth_content'].get()['shuffle_cells']
            self.shuffle_cells_portion_selector = \
                config_selectors['html']['synth_content'].get()['shuffle_cells'].get()["portion"]
            self.mix_thead_tbody_switch = config_selectors['html']['synth_content'].get()['corpus']['mix_thead_tbody']

    def _sample_cell_text(self, thead_or_tbody='tbody', mix_thead_tbody=False):
        if mix_thead_tbody:
            thead_or_tbody = ["tbody", "thead"][np.random.randint(0, 2)]
        if thead_or_tbody == "tbody":
            corpus_type = self.tbody_corpus_selector.select()['name']
        else:
            corpus_type = self.thead_corpus_selector.select()['name']
        corpus = self.corpus_dict[thead_or_tbody][corpus_type]
        return corpus.sample()['text']

    def _sample_global_color_mode(self):
        return self.global_color_mode.select()

    def _sample_dark_color(self):
        return self.dark_colors.select()

    def _sample_light_color(self):
        return self.light_colors.select()

    def _sample_global_table_wrapper(self):
        self._sample_table_wrapper_background()

    def _sample_table_wrapper_background(self):
        # background
        color_mode = self.meta['color_mode']

        self.global_style["table"]["color"] = self._sample_fg_color(color_mode)
        if self.meta['background_config'] == 'gradient':
            gradient_type = self.gradient_bg['type'].select()
            angle = self.gradient_bg['angle'].select()
            num_colors = self.gradient_bg['num_colors'].select()
            # todo:
            # random_stop_position = self.gradient_bg['random_stop_position'].on()

            gd_colors = []
            for i in range(num_colors):
                gd_colors.append(self._sample_bg_color(color_mode))

            if gradient_type == "conic":
                bg_gradient = "{}-gradient(from {}deg, {})".format(gradient_type, angle, ", ".join(gd_colors))
            elif gradient_type == "radial":
                bg_gradient = "{}-gradient({})".format(gradient_type, ", ".join(gd_colors))
            else:
                bg_gradient = "{}-gradient({}deg, {})".format(gradient_type, angle, ", ".join(gd_colors))
            self.global_style["#table_wrapper"]["background"] = bg_gradient
        elif self.meta['background_config'] == 'empty':
            if color_mode == "dark":
                self.global_style["#table_wrapper"]["background-color"] = 'black'
                self.global_style["table"]["color"] = 'white'
            else:
                self.global_style["#table_wrapper"]["background-color"] = 'white'
                self.global_style["table"]["color"] = 'black'
        elif self.meta['background_config'] == 'solid':
            self.global_style["#table_wrapper"]["background-color"] = self._sample_bg_color(color_mode)

    def _sample_table_background(self):
        # background
        color_mode = self.meta['color_mode']

        background_config = self.config_selectors['style']['global']['absolute']['table']['background'].select()
        self.meta['table_background_config'] = background_config['name']
        # self.global_style["table"]["color"] = self._sample_fg_color(color_mode)

        thead = False
        if self.meta['table_background_config'] not in ['empty', 'solid']:
            thead = background_config['config']['thead'].on() and self.meta['has_thead']

        if self.meta['table_background_config'] == 'striped':
            dark_line = background_config['config']['dark_line'].select()
            light_line = "even" if dark_line == "odd" else "odd"
            dark_color = self._sample_dark_color()
            light_color = self._sample_light_color()
            self.global_style["tbody tr:nth-child({})".format(dark_line)]["background-color"] = dark_color
            self.global_style["tbody tr:nth-child({})".format(dark_line)]["color"] = light_color
            self.global_style["tbody tr:nth-child({})".format(light_line)]["background-color"] = light_color
            self.global_style["tbody tr:nth-child({})".format(light_line)]["color"] = dark_color
            if thead:
                if self.meta['nums_head_row'] % 2 == 0:
                    self.global_style["thead tr:nth-child({})".format(dark_line)]["background-color"] = dark_color
                    self.global_style["thead tr:nth-child({})".format(dark_line)]["color"] = light_color
                    self.global_style["thead tr:nth-child({})".format(light_line)]["background-color"] = light_color
                    self.global_style["thead tr:nth-child({})".format(light_line)]["color"] = dark_color
                else:
                    self.global_style["thead tr:nth-child({})".format(dark_line)]["background-color"] = light_color
                    self.global_style["thead tr:nth-child({})".format(dark_line)]["color"] = dark_color
                    self.global_style["thead tr:nth-child({})".format(light_line)]["background-color"] = dark_color
                    self.global_style["thead tr:nth-child({})".format(light_line)]["color"] = light_color
        elif self.meta['table_background_config'] == 'striped_same_color_mode':
            bg_color_odd = self._sample_bg_color(color_mode)
            bg_color_even = self._sample_bg_color(color_mode)
            font_color = self._sample_fg_color(color_mode)
            self.global_style["tbody tr:nth-child(odd)"]["background-color"] = bg_color_odd
            self.global_style["tbody tr:nth-child(even)"]["background-color"] = bg_color_even
            self.global_style["tbody tr:nth-child(odd)"]["color"] = font_color
            self.global_style["tbody tr:nth-child(even)"]["color"] = font_color
            if thead:
                if self.meta['nums_head_row'] % 2 == 0:
                    self.global_style["thead tr:nth-child(odd)"]["background-color"] = bg_color_odd
                    self.global_style["thead tr:nth-child(even)"]["background-color"] = bg_color_even
                else:
                    self.global_style["thead tr:nth-child(odd)"]["background-color"] = bg_color_even
                    self.global_style["thead tr:nth-child(even)"]["background-color"] = bg_color_odd
                self.global_style["thead tr:nth-child(odd)"]["color"] = font_color
                self.global_style["thead tr:nth-child(even)"]["color"] = font_color
        elif self.meta['table_background_config'] == 'multi_color':
            font_color = self._sample_fg_color(color_mode)
            if self.meta['has_thead']:
                max_row = self.meta['nums_row'] - self.meta['nums_head_row']
            else:
                max_row = self.meta['nums_row']
            for i in range(1, max_row + 1):
            # for i in range(1, self.meta['nums_row'] - self.meta['nums_head_row'] + 1):
                self.global_style["tbody tr:nth-child({})".format(i)]["background-color"] = self._sample_bg_color(
                    color_mode)
                self.global_style["tbody tr:nth-child({})".format(i)]["color"] = font_color
            if self.meta['has_thead']:
                for i in range(1, self.meta['nums_head_row'] + 1):
                    self.global_style["thead tr:nth-child({})".format(i)]["background-color"] = self._sample_bg_color(
                        color_mode)
                    self.global_style["thead tr:nth-child({})".format(i)]["color"] = font_color
        elif self.meta['table_background_config'] == 'solid':
            self.global_style["table"]["background-color"] = self._sample_bg_color(color_mode)
            self.global_style["table"]["color"] = self._sample_fg_color(color_mode)

    def _sample_global_thead_outline(self):
        thead_outline_type = self.config_selectors['style']['global']['absolute']['thead']['outline'].select()
        self.meta['thead_outline_type'] = thead_outline_type

        if thead_outline_type == "all":
            self._set_global_four_borders('thead')
        elif thead_outline_type == "top_bottom":
            self._set_global_top_bottom_borders('thead')
        elif thead_outline_type == "bottom":
            self._set_global_border('thead', 'bottom')

    def _sample_global_inner_border(self, css_selector):
        inner_border_type = self.config_selectors['style']['global']['absolute'][css_selector][
            'inner_border'].select()
        self.meta[css_selector + '_inner_border_type'] = inner_border_type
        if inner_border_type == "all":
            self._set_inner_border_row(css_selector)
            self._set_inner_border_col(css_selector)
        elif inner_border_type == "row":
            self._set_inner_border_row(css_selector)
        elif inner_border_type == "col":
            self._set_inner_border_col(css_selector)
            if css_selector == "thead" and self.meta['nums_head_row'] > 1:
                self._set_inner_border_row(css_selector)
        if self.meta['has_thead'] and css_selector == "tbody" and inner_border_type != "empty":
            if self.meta['thead_outline_type'] == "empty":
                self._set_global_border('thead', 'bottom')
            if self.meta['nums_head_row'] > 1 and self.meta['thead_inner_border_type'] in ["col", "empty"]:
                self._set_inner_border_row("thead")

    def _set_inner_border_row(self, css_selector):
        tr_elements = self.meta['html_bs'].find(css_selector).find_all("tr")
        for ridx in range(len(tr_elements) - 1):
            for cidx, td_element in enumerate(tr_elements[ridx].find_all("td")):
                tmp_ridx = ridx
                if self.meta['has_thead']:
                    tmp_ridx = self.meta['nums_head_row'] - ridx

                if td_element.has_attr('rowspan') and int(td_element['rowspan']) == tmp_ridx:
                    continue
                self._set_global_border('{} tr:nth-child({}) td:nth-child({})'.format(css_selector, ridx + 1, cidx + 1),
                                        'bottom')

    def _set_inner_border_col(self, css_selector):
        tr_tags = self.meta['html_bs'].find(css_selector).find_all("tr")
        if self.meta['span']:
            table_row_span_map = np.full((len(tr_tags), self.meta['nums_col']), False)
        for ridx, tr_element in enumerate(tr_tags):
            real_cidx = 0
            td_tags = tr_element.find_all("td")
            for cidx, td_tag in enumerate(td_tags):
                if self.meta['span']:
                    while table_row_span_map[ridx][real_cidx]:
                        real_cidx += 1

                    has_row_span = td_tag.has_attr('rowspan')
                    has_col_span = td_tag.has_attr('colspan')
                    if has_row_span and has_col_span:
                        table_row_span_map[ridx:ridx + int(td_tag['rowspan']),
                        real_cidx:real_cidx + int(td_tag['colspan'])] = True
                    elif has_row_span:
                        table_row_span_map[ridx:ridx + int(td_tag['rowspan']), real_cidx] = True
                    elif has_col_span:
                        table_row_span_map[ridx, real_cidx:real_cidx + int(td_tag['colspan'])] = True

                    real_cidx += int(td_tag['colspan']) if has_col_span else 1

                if cidx == len(td_tags) - 1:
                    if self.meta['span'] and real_cidx < self.meta['nums_col'] and all(
                            [table_row_span_map[ridx][inner_cidx] for inner_cidx in
                             range(real_cidx, self.meta['nums_col'])]):
                        self._set_global_border(
                            '{} tr:nth-child({}) td:nth-child({})'.format(css_selector, ridx + 1, cidx + 1),
                            'right')
                else:
                    self._set_global_border(
                        '{} tr:nth-child({}) td:nth-child({})'.format(css_selector, ridx + 1, cidx + 1),
                        'right')

    def _sample_global_thead(self):
        self._sample_global_thead_outline()
        self._sample_global_inner_border("thead")

    def _sample_global_tbody(self):
        self._sample_global_inner_border("tbody")

    def _sample_font(self, selector):
        font_path = os.path.abspath(self.font.sample({"text": self.content_text})["path"])
        print(self.content_text, font_path)
        font_family = str(uuid.uuid4())
        font_face = {
            'font-family': font_family,
            'src': 'url("{}") format("truetype")'.format(font_path),
            'font-weight': 'normal'
        }
        if '@font-face' not in self.global_style:
            self.global_style['@font-face'] = []
        self.global_style['@font-face'].append(font_face)
        self.global_style[selector]['font-family'] = font_face['font-family']
        self.meta['{}_font'.format(selector)] = font_face['src']

    def _sample_bg_color(self, color_mode):
        if color_mode == "dark":
            return self.dark_colors.select()
        elif color_mode == "light":
            return self.light_colors.select()

    def _sample_fg_color(self, color_mode):
        if color_mode == "dark":
            return self.light_colors.select()
        elif color_mode == "light":
            return self.dark_colors.select()

    def _set_global_border(self, css_selector, lrtb):
        self.global_style[css_selector]['border-{}-style'.format(lrtb)] = self.meta['table_border_style']
        self.global_style[css_selector]['border-{}-width'.format(lrtb)] = self.meta['table_border_width']
        self.global_style[css_selector]['border-{}-color'.format(lrtb)] = self.meta['table_outline_border_color']

    def _set_global_four_borders(self, css_selector):
        self._set_global_border(css_selector, "top")
        self._set_global_border(css_selector, "bottom")
        self._set_global_border(css_selector, "left")
        self._set_global_border(css_selector, "right")

    def _set_global_left_right_borders(self, css_selector):
        self._set_global_border(css_selector, "left")
        self._set_global_border(css_selector, "right")

    def _set_global_top_bottom_borders(self, css_selector):
        self._set_global_border(css_selector, "top")
        self._set_global_border(css_selector, "bottom")

    def _sample_table_outline(self):
        table_outline_type = self.config_selectors['style']['global']['absolute']['table']['outline'].select()
        self.meta['table_outline_type'] = table_outline_type
        border_color = self._sample_fg_color(self.meta['color_mode'])
        self.meta['table_outline_border_color'] = border_color
        # if table_outline_type == "empty":
        #     self.global_style['table']['border-top-style'] = 'hidden'
        #     self.global_style['table']['border-bottom-style'] = 'hidden'
        #     self.global_style['table']['border-left-style'] = 'hidden'
        #     self.global_style['table']['border-right-style'] = 'hidden'
        if table_outline_type == "all":
            self._set_global_four_borders('table')
        elif table_outline_type == "top_bottom":
            self._set_global_top_bottom_borders('table')

    def _sample_global_table(self):
        # global font
        self.meta['table_font'] = self.config_selectors['style']['global']['absolute']['table']['font'].on()
        if self.meta['table_font']:
            self._sample_font('table')

        # table image full size
        self.meta['table_full_size'] = self.config_selectors['style']['global']['absolute']['table']['full_size'].on()

        # table image aspect_ratio
        self.meta['table_aspect_ratio'] = self.config["style"]['global']['absolute']['table']["aspect_ratio"]

        # table border width
        self.meta['table_border_width'] = self.config_selectors['style']['global']['absolute']['table'][
            'border_width'].select()

        # table border style
        self.meta['table_border_style'] = self.config_selectors['style']['global']['absolute']['table'][
            'border_style'].select()

        self._sample_table_outline()
        if self.meta['background_config'] != 'paper':
            self._sample_table_background()

    def _sample_global_vars(self):
        # global color mode
        self.meta['color_mode'] = self._sample_global_color_mode()

        background_config = self.config_selectors['style']['global']['absolute']['table_wrapper']['background'].select()
        self.meta['background_config'] = background_config['name']
        if self.meta['background_config'] == "paper":
            self.meta['color_mode'] = "light"

    def sample_styles(self):
        # static style
        self.global_style = defaultdict(dict)
        self.global_style['a']['color'] = 'inherit'
        self.global_style['#table_wrapper']["display"] = "inline-block"

        # sample global vars
        self._sample_global_vars()

        # sample global css
        css_selectors = self.config_selectors['style']['global']['css']
        self._set_css_to_global_style(css_selectors)

        # sample table wrapper style
        self._sample_global_table_wrapper()

        # sample table style
        self._sample_global_table()

        if self.meta['has_thead']:
            self._sample_global_thead()
        self._sample_global_tbody()

        # local style
        self.sample_local_styles()

        return self.global_style

    def _set_css_to_global_style(self, css_selectors, ):
        for css_selector in css_selectors:
            for css_key in css_selectors[css_selector]:
                selector = css_selectors[css_selector][css_key]
                value = selector.select()
                if value is None:
                    continue
                self.global_style[css_selector][css_key] = value

    def _set_text_style(self, bs, bs_element, text, text_config, color_mode):
        local_config = self.config_selectors['style']['local'].get()
        text_id = str(uuid.uuid4())
        span_tag = bs.new_tag("span", id="text_{}".format(text_id))
        span_tag.append(text)
        bs_element.append(span_tag)
        selectors = local_config['css']['text']
        global_style_key = "#text_{}".format(text_id)
        for css_selector in selectors:
            css_val = selectors[css_selector]
            val = css_val.select()
            if val is None:
                continue
            self.global_style[global_style_key][css_selector] = val

        # color
        use_fg = text_config['config']['fg_color_change'].on()
        use_bg = text_config['config']['bg_color_change'].on()
        if ('table_background_config' in self.meta and self.meta[
            'table_background_config'] == 'striped' and use_fg and use_bg and self.meta[
                'background_config'] != 'paper') or ('table_background_config' in self.meta and self.meta[
            'table_background_config'] != 'striped' and use_fg):
            self.global_style[global_style_key]['color'] = self._sample_fg_color(color_mode)

            if use_bg and self.meta['background_config'] != 'paper':
                self.global_style[global_style_key]['background-color'] = self._sample_bg_color(color_mode)

        # font
        if text_config['config']['font']:
            self._sample_font(global_style_key)

        # font-size
        font_size_scale = local_config['relative']['td']['text']['font_size'].select()
        if font_size_scale:
            font_size = int(
                round(float(self.global_style['table']['font-size'].split("px")[0]) * font_size_scale))
            font_size = min(self.max_font_size, max(self.min_font_size, font_size))
            self.global_style[global_style_key]['font-size'] = str(font_size) + "px"

    def _set_local_text_styles(self, bs_element, color_mode):
        local_config = self.config_selectors['style']['local'].get()
        text_config = local_config['absolute']['td'].get()['text'].get().select()
        word_or_char = text_config['name']
        bs = self.meta['html_bs']
        text = remove_tags("".join([str(tag) for tag in bs_element.contents])).strip()
        bs_element.clear()

        if word_or_char == 'word':
            words = text.split(" ")
            change_word_ratio = text_config['config']['words'].select()
            change_word_cnt = math.ceil(len(words) * change_word_ratio)
            if change_word_cnt > len(words):
                change_word_cnt = len(words)
            change_word_indexes = random.sample(list(range(len(words))), change_word_cnt)
            for i, word in enumerate(words):
                if i > 0:
                    bs_element.append(" ")
                if i in change_word_indexes:
                    self._set_text_style(bs, bs_element, word, text_config, color_mode)
                else:
                    bs_element.append(word)
                if i < len(words):
                    bs_element.append(" ")
        elif word_or_char == "char":
            char_length_ratio = text_config['config']['length'].select()
            char_length = math.ceil(len(text) * char_length_ratio)
            if char_length > len(text):
                char_length = len(text)
            start_idx = np.random.randint(0, len(text) - char_length + 1)
            if start_idx > 0:
                bs_element.append(text[:start_idx])
            self._set_text_style(bs, bs_element, text[start_idx:start_idx + char_length], text_config, color_mode)
            if start_idx + char_length < len(text):
                bs_element.append(text[start_idx + char_length:])

    def _change_global_border(self, css_selector, border_key, border_value):
        k = "border-{}".format(border_key)
        if k in self.global_style[css_selector]:
            self.global_style[css_selector][k] = border_value

        for ltrb in ['left', 'top', 'right', 'bottom']:
            k = "border-{}-{}".format(ltrb, border_key)
            if k in self.global_style[css_selector]:
                self.global_style[css_selector][k] = border_value

        if css_selector in ["thead", "tbody"]:
            tr_elements = self.meta['html_bs'].find(css_selector).find_all("tr")
            for ridx in range(1, len(tr_elements)):
                k = '{} tr:nth-child({}) td'.format(css_selector, ridx)
                if k not in self.global_style:
                    break
                self.global_style[k]['border-bottom-{}'.format(border_key)] = border_value

            for ridx, tr_element in enumerate(tr_elements):
                is_break = False
                for cidx in range(1, len(tr_element.find_all("td"))):
                    k = '{} tr:nth-child({}) td:nth-child({})'.format(css_selector, ridx + 1, cidx)
                    if k not in self.global_style:
                        is_break = True
                        break
                    self.global_style[k]['border-right-{}'.format(border_key)] = border_value
                if is_break:
                    break

    def _set_global_border_color(self, css_selector, border_color, color_mode):
        if "border-width" in self.global_style[css_selector] or "border-style" in self.global_style[css_selector]:
            self.global_style[css_selector]["border-color"] = border_color if border_color else self._sample_fg_color(
                color_mode)

        for ltrb in ['left', 'top', 'right', 'bottom']:
            k1 = "border-{}-{}".format(ltrb, "width")
            k2 = "border-{}-{}".format(ltrb, "style")
            if k1 in self.global_style[css_selector] or k2 in self.global_style[css_selector]:
                self.global_style[css_selector][
                    "border-{}-color".format(ltrb)] = border_color if border_color else self._sample_fg_color(
                    color_mode)

    def _set_local_css_styles(self, config_key, css_selector_name, parent_color_mode, bs_element=None):
        local_config = self.config_selectors['style']['local'].get()

        color_mode = parent_color_mode
        if local_config['absolute'][config_key].on():
            # css
            selectors = local_config['css'][config_key]
            for css_selector in selectors:
                css_val = selectors[css_selector]
                val = css_val.select()
                if val is None:
                    continue
                self.global_style[css_selector_name][css_selector] = val

            # absolute
            absolute_config = local_config['absolute'][config_key].get()

            # color
            if 'color_mode' in absolute_config:
                absolute_color_mode = absolute_config['color_mode'].select()
                if absolute_color_mode:
                    color_mode = absolute_color_mode
                    if self.meta['background_config'] == 'paper':
                        color_mode = "light"
                    self.meta[css_selector_name + "_color_mode"] = color_mode
                    # border color
                    border_color = self._sample_fg_color(color_mode)
                    if config_key in ["thead", "tbody"]:
                        self._change_global_border(css_selector_name, 'color', border_color)
                    else:
                        same_color = absolute_config['same_color'].on()
                        if not same_color:
                            border_color = None
                        self._set_global_border_color(css_selector_name, border_color, color_mode)
                    # font color
                    self.global_style[css_selector_name]['color'] = self._sample_fg_color(color_mode)
                    # bg color
                    if self.meta['background_config'] != 'paper':
                        self.global_style[css_selector_name]['background-color'] = self._sample_bg_color(color_mode)

            if config_key in ["thead", "tbody"]:
                # border width
                if 'border_width' in absolute_config:
                    border_width = absolute_config['border_width'].select()
                    if border_width:
                        self._change_global_border(css_selector_name, 'width', border_width)

                # border style
                if 'border_style' in absolute_config:
                    border_style = absolute_config['border_style'].select()
                    if border_style:
                        self._change_global_border(css_selector_name, 'style', border_style)

            # font
            if 'font' in absolute_config:
                self.meta[css_selector_name + "_font"] = absolute_config['font'].on()
                if self.meta[css_selector_name + "_font"]:
                    self._sample_font(css_selector_name)

            # td: text vertical
            if config_key == "td" and absolute_config['text_vertical'].on():
                max_text_length = absolute_config['text_vertical'].get()[
                    "max_text_length"].select()
                ignore_number = absolute_config['text_vertical'].get()[
                    "ignore_number"].on()

                if 1 < len(bs_element.text) <= max_text_length and (
                        not ignore_number or (ignore_number and all(not c.isdigit() for c in bs_element.text))):
                    self.global_style[css_selector_name]['text-orientation'] = 'upright'
                    self.global_style[css_selector_name]['writing-mode'] = 'vertical-rl'

            # relative
            if 'font_size' in local_config['relative'][config_key]:
                self.meta[css_selector_name + "_font_size_scale"] = local_config['relative'][config_key][
                    'font_size'].select()
                if self.meta[css_selector_name + "_font_size_scale"]:
                    font_size_scale = self.meta[css_selector_name + "_font_size_scale"]
                    font_size = int(
                        round(float(self.global_style['table']['font-size'].split("px")[0]) * font_size_scale))
                    font_size = min(self.max_font_size, max(self.min_font_size, font_size))
                    self.global_style[css_selector_name]['font-size'] = str(font_size) + "px"
                    self.meta[css_selector_name + "_font_size_scale"] = font_size_scale
                    self.meta[css_selector_name + "_font_size"] = font_size

        if config_key == "td" and local_config['absolute']['td'].get()['text'].on():
            self._set_local_text_styles(bs_element, color_mode)
        return color_mode

    def _sample_local_hierical_tag_styles(self, parent_selector):
        thead_tag = self.meta['html_bs'].find(parent_selector)
        color_mode = self.meta['color_mode']
        color_mode = self._set_local_css_styles(parent_selector, parent_selector, parent_color_mode=color_mode)
        for row_idx, tr_tag in enumerate(thead_tag.find_all("tr")):
            color_mode = self._set_local_css_styles('tr',
                                                    "{} tr:nth-child({})".format(parent_selector, row_idx + 1),
                                                    bs_element=tr_tag, parent_color_mode=color_mode)
            for col_idx, td_tag in enumerate(tr_tag.find_all("td")):
                self._set_local_css_styles('td',
                                           "{} tr:nth-child({}) td:nth-child({})".format(parent_selector,
                                                                                         row_idx + 1, col_idx + 1),
                                           bs_element=td_tag, parent_color_mode=color_mode)

    def sample_local_styles(self):
        if not self.config_selectors['style']['local'].on():
            return

        if 'html_bs' not in self.meta:
            self.meta['html_bs'] = BeautifulSoup(self.meta['html'], 'html.parser')

        if self.meta['has_thead']:
            self._sample_local_hierical_tag_styles("thead")

        self._sample_local_hierical_tag_styles("tbody")

        # thead_and_tbody_list = ["tbody"]
        # if self.meta['has_thead']:
        #     self._set_local_css_styles('thead', 'thead')
        #     thead_and_tbody_list.append("thead")
        # self._set_local_css_styles('tbody', 'tbody')
        #
        # if 'html_bs' not in self.meta:
        #     self.meta['html_bs'] = BeautifulSoup(self.meta['html'], 'html.parser')
        # for thead_or_tbody in thead_and_tbody_list:
        #     thead_or_tbody_element = self.meta['html_bs'].find(thead_or_tbody)
        #     for row_idx, tr_tag in enumerate(thead_or_tbody_element.find_all("tr")):
        #         # for row_idx in range(1, self.meta['nums_row'] + 1):
        #         self._set_local_css_styles('tr', "{} tr:nth-child({})".format(thead_or_tbody, row_idx + 1),
        #                                    bs_element=tr_tag)
        #         for col_idx, td_tag in enumerate(tr_tag.find_all("td")):
        #             # for col_idx in range(1, self.meta['nums_col'] + 1):
        #             self._set_local_css_styles('td',
        #                                        "{} tr:nth-child({}) td:nth-child({})".format(thead_or_tbody,
        #                                                                                      row_idx + 1, col_idx + 1),
        #                                        bs_element=td_tag)
        # for row_idx, tr_tag in enumerate(self.meta['html_bs'].find_all("tr")):
        #     # for row_idx in range(1, self.meta['nums_row'] + 1):
        #     self._set_local_css_styles('tr', "tr:nth-child({})".format(row_idx + 1), tr_tag)
        #     for col_idx, td_tag in enumerate(tr_tag.find_all("td")):
        #         # for col_idx in range(1, self.meta['nums_col'] + 1):
        #         self._set_local_css_styles('td',
        #                                    "tr:nth-child({}) td:nth-child({})".format(row_idx + 1, col_idx + 1),
        #                                    td_tag)

        self.meta['html'] = convert_bs_to_html_string(self.meta['html_bs'])

    def sample(self, meta=None):
        # synth structure config
        structure_config = self.config_selectors['html']['structure'].select()
        synth_structure = structure_config['name'] == 'synth_structure'
        self.meta['structure_type'] = structure_config['name']
        if synth_structure:
            # synth structure
            synth_structure_config = structure_config['config']
            self.synth_structure_config = synth_structure_config
            self.meta['nums_row'] = synth_structure_config['nums_row'].select()
            self.meta['nums_col'] = synth_structure_config['nums_col'].select()

            self.meta['span'] = synth_structure_config['span'].on()
            self.meta['add_thead'] = synth_structure_config['thead'].on()
            if self.meta['add_thead']:
                self.meta['thead_rows'] = synth_structure_config['thead'].get()['rows'].select()
                self.meta['nums_head_row'] = self.meta['thead_rows']
            self.meta['has_thead'] = self.meta['add_thead']
            self.meta['mix_thead_tbody'] = self.mix_thead_tbody_switch.on()
            self._synth_structure_and_content()
        else:
            # static html
            html_result = self._sample_html_path()
            if html_result is False:
                return self.sample()
            html_path, html_json = html_result
            if 'html_bs' not in self.meta:
                bs = BeautifulSoup(html_json['html'], 'html.parser')
                self.meta['html_bs'] = bs
            self.meta['html_path'] = html_path
            html = html_json['html'].strip()
            # insert tbody
            if not self.meta['html_bs'].find("tbody"):
                html = html_util.insert_tbody_tag(self.meta['html_bs'])
            self.meta['has_thead'] = True if self.meta['html_bs'].find("thead") else False
            if self.meta['has_thead']:
                if 'html_bs' not in self.meta:
                    self.meta['html_bs'] = BeautifulSoup(html_json['html'], 'html.parser')
                self.meta['nums_head_row'] = len(self.meta['html_bs'].find("thead").find_all("tr"))
            self.meta['html'] = html
            self.meta['original_html'] = html
            self.meta['nums_col'] = html_json['nums_col']
            self.meta['nums_row'] = html_json['nums_row']
            self.meta['span'] = html_json['has_span']

        if 'html_bs' not in self.meta:
            self.meta['html_bs'] = BeautifulSoup(self.meta['html'], 'html.parser')

        if self.max_empty_cell_ratio:
            num_empty_tds = 0
            tds = self.meta['html_bs'].find_all("td")
            for td in tds:
                if not td.text.strip():
                    num_empty_tds += 1
            if num_empty_tds / len(tds) > self.max_empty_cell_ratio:
                return self.sample()

        # synth config
        synth_content = self.config_selectors['html']['synth_content'].on()
        self.meta['synth_content'] = False if synth_structure else synth_content
        if self.meta['synth_content']:
            self.meta['shuffle_cells'] = self.shuffle_cells_switch.on()
            self.meta['mix_thead_tbody'] = self.mix_thead_tbody_switch.on()
            self._synth_content()

        self.content_text = "".join(set(html_util.remove_white_spaces(self.meta['html_bs'].get_text())))

        # styling
        self.meta['global_style'] = self.sample_styles()

        # global relative styles
        relative_style = defaultdict(dict)
        relative_style_config = self.config_selectors['style']['global']['relative']
        for selector in relative_style_config:
            for key in relative_style_config[selector]:
                relative_style[selector][key] = relative_style_config[selector][key].select()
        self.meta['relative_style'] = relative_style

        # etc
        self.meta['tmp_path'] = self.config_selectors['html']['tmp_path'].select()

        self.meta['max_image_height'] = self.max_image_height
        self.meta['max_image_width'] = self.max_image_width

        self.meta['max_big_image_ratio'] = self.max_big_image_ratio
        self.meta['max_big_image_ratio_when_less_cells'] = self.max_big_image_ratio_when_less_cells
        self.meta['num_less_cell'] = self.num_less_cell

        self.meta['max_empty_cell_ratio'] = self.max_empty_cell_ratio

        return self.meta

    def _sample_html_path(self):
        try_cnt = 0
        while True:
            try_cnt += 1
            if try_cnt >= 100:
                print("Failed to find the html file with that condition.")
                return False

            if self.html_path_shuffle:
                html_json_path, _, _ = self.html_path_selector.select()
            else:
                html_json_path = self.html_path_selector.get(0, self.html_file_idx)
                if len(self.html_path_selector.paths[0]) <= self.html_file_idx + 1:
                    self.html_file_idx = 0
                else:
                    self.html_file_idx += 1

            html_json = json.load(open(html_json_path), encoding='utf-8')

            if self.html_path_shuffle:
                if self.html_charset:
                    bs = BeautifulSoup(html_json['html'], 'html.parser')
                    self.meta['html_bs'] = bs
                    if not self.html_charset.check_charset(bs.text):
                        continue
                if self.min_cols > html_json['nums_col'] or self.max_cols < html_json['nums_col']:
                    continue
                if self.min_rows > html_json['nums_row'] or self.max_rows < html_json['nums_row']:
                    continue
                has_span = self.has_span.on()
                if has_span != html_json['has_span']:
                    continue

                if has_span:
                    if self.has_row_span.on() and not html_json['has_row_span']:
                        continue
                    if self.has_col_span.on() and not html_json['has_col_span']:
                        continue
            return html_json_path, html_json

    def _synth_structure_and_content(self):
        if self.meta['span']:
            span_table = np.full((self.meta['nums_row'], self.meta['nums_col']), False)

        tags = ["<table>"]
        add_thead = self.meta['add_thead']
        if add_thead:
            thead_rows = self.meta['thead_rows']
            if thead_rows > self.meta['nums_row'] - 1:
                thead_rows = self.meta['nums_row'] - 1
        for row in range(self.meta['nums_row']):
            if add_thead:
                if row == 0:
                    tags.append("<thead>")
                elif row == thead_rows:
                    tags.append("<tbody>")
                is_head = row < thead_rows
            else:
                if row == 0:
                    tags.append("<tbody>")
                is_head = False
            tags.append("<tr>")
            for col in range(self.meta['nums_col']):
                if self.meta['span']:
                    if span_table[row][col]:
                        continue
                    spans = []
                    if self.synth_structure_config['span'].get()['row_span'].on():
                        if is_head:
                            max_row_span = thead_rows - row
                        else:
                            max_row_span = self.meta['nums_row'] - row

                        for row_span_idx in range(1, max_row_span):
                            if span_table[row + row_span_idx][col]:
                                max_row_span = row_span_idx - 1
                                break

                        if max_row_span > 1:
                            row_span = np.random.randint(2, max_row_span + 1)
                            spans.append(' rowspan="{}"'.format(row_span))
                            span_table[row:row + row_span, col] = True
                    if self.synth_structure_config['span'].get()['col_span'].on():
                        max_col_span = self.meta['nums_col'] - col
                        for col_span_idx in range(1, max_col_span):
                            if span_table[row][col + col_span_idx]:
                                max_col_span = col_span_idx - 1
                                break

                        if max_col_span > 1:
                            col_span = np.random.randint(2, max_col_span + 1)
                            spans.append(' colspan="{}"'.format(col_span))
                            span_table[row, col:col + col_span] = True
                    if len(spans) == 2:
                        span_table[row:row + row_span, col:col + col_span] = True
                    span_attr = "".join(spans)
                    tags.append("<td{}>".format(span_attr))
                else:
                    tags.append("<td>")
                if not self.empty_cell_switch.on():
                    tags.append(self._sample_cell_text("thead" if is_head else "tbody", self.meta['mix_thead_tbody']))
                tags.append("</td>")
            tags.append("</tr>")
            if add_thead and thead_rows == row + 1:
                tags.append("</thead>")
            if row == self.meta['nums_row'] - 1:
                tags.append("</tbody>")
        tags.append("</table>")
        self.meta['html'] = "".join(tags)

    def _swap_cells(self, bs, first_td_tag, second_td_tag, bold=False, idx=None):
        first_text = "".join([str(tag) for tag in first_td_tag.contents])
        second_text = "".join([str(tag) for tag in second_td_tag.contents])
        first_text = remove_tags(first_text).strip()
        second_text = remove_tags(second_text).strip()
        self.meta["swap_cell_{}".format(idx)] = {
            "first_text": first_text,
            "second_text": second_text,
        }
        if bold:
            first_btag = bs.new_tag("b")
            first_btag.string = first_text
            second_btag = bs.new_tag("b")
            second_btag.string = second_text
            first_td_tag.clear()
            second_td_tag.clear()
            first_td_tag.append(second_btag)
            second_td_tag.append(first_btag)
        else:
            first_td_tag.string = second_text
            second_td_tag.string = first_text

    def _shuffle_cells(self, bs, element, is_thead=False):
        td_tags = element.find_all("td")
        swap_cnt = math.ceil(len(td_tags) * self.shuffle_cells_portion_selector.select())
        if swap_cnt % 2 != 0:
            swap_cnt += 1
        if len(td_tags) < swap_cnt:
            swap_cnt -= 2
        if swap_cnt < 1:
            return

        shuffle_td_tags = random.sample(td_tags, swap_cnt)

        if is_thead:
            is_bold = self.thead_bold_switch.on()
            self.meta['shuffle_thead_is_bold'] = is_bold
            self.meta['shuffle_thead_swap_cnt'] = swap_cnt
        else:
            is_bold = False
            self.meta['shuffle_swap_cnt'] = swap_cnt
        for i in range(0, len(shuffle_td_tags), 2):
            self._swap_cells(bs, shuffle_td_tags[i], shuffle_td_tags[i + 1], is_bold,
                             ("thead" if is_thead else "") + str(i))

    def _synth_content(self):
        if 'html_bs' not in self.meta:
            self.meta['html_bs'] = BeautifulSoup(self.meta['html'], 'html.parser')
        bs = self.meta['html_bs']
        if self.meta['shuffle_cells']:
            if self.meta['mix_thead_tbody']:
                self._shuffle_cells(bs, bs)
            else:
                thead_element = bs.find("thead")
                if thead_element:
                    self._shuffle_cells(bs, thead_element, True)

                tbody_element = bs.find("tbody")
                if tbody_element:
                    self._shuffle_cells(bs, tbody_element)
        else:
            thead_bold = self.thead_bold_switch.on()
            self.meta['synth_content_thead_bold'] = thead_bold
            for thead_or_tbody in ["thead", "tbody"]:
                if not thead_or_tbody:
                    continue
                thead_or_tbody_tag = bs.find(thead_or_tbody)
                for td in thead_or_tbody_tag.find_all("td"):
                    if self.synth_cell_switch.on():
                        if self.empty_cell_switch.on():
                            td.string = ""
                        else:
                            cell_text = self._sample_cell_text(thead_or_tbody, self.meta['mix_thead_tbody'])
                            if thead_or_tbody == "thead" and thead_bold:
                                btag = bs.new_tag("b")
                                btag.string = cell_text
                                td.clear()
                                td.append(btag)
                            else:
                                td.string = cell_text
        self.meta['html'] = convert_bs_to_html_string(bs)

    def apply(self, layers, meta=None):
        if meta is None:
            meta = {}
        self.meta = meta
        self.sample()

        if self.meta['background_config'] == 'paper':
            paper = self.paper
            self.meta['global_style']["table"]["color"] = self._sample_dark_color()
        else:
            paper = None

        self.meta["effect_config"] = self.config['effect']
        for layer in layers:
            # rendering
            ret = layer.render_table(paper=paper, meta=self.meta)
            if not ret:
                return self.apply(layers)
