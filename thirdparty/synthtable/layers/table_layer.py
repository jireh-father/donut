import traceback
from synthtiger.layers.layer import Layer
from selenium import webdriver
from PIL import Image
from wand.image import Image as WandImage
import uuid
import os
import numpy as np
import sys
from selenium.webdriver.common.by import By

from utils import image_util
from utils.selector import parse_config
import components


class TableLayer(Layer):
    def __init__(self, size, selenium_driver):
        self.table_size = size
        self.html = None
        self.meta = None
        self.global_style = {}
        self.selenium_driver = selenium_driver

    def _convert_global_style_to_css(self):
        css_list = []
        for selector in self.global_style:
            if selector == '@font-face':
                for font_face_dict in self.global_style[selector]:
                    styles = []
                    for key in font_face_dict:
                        styles.append("{}: {};".format(key, font_face_dict[key]))
                    css_list.append(selector + " { " + "".join(styles) + " }")
            else:
                styles = []
                for key in self.global_style[selector]:
                    value = self.global_style[selector][key]
                    styles.append("{}: {};".format(key, value))

                css_list.append(selector + " { " + "".join(styles) + " }")

        return " ".join(css_list)

    def _write_html_file(self, html_path):
        html_template = """
        <html>
        <head>
            <meta charset="UTF-8">
             <style>
             {}
            </style>
        </head>
        <body>
            <div id="table_wrapper">
                {}
            </div>
        </body>
        </html>
        """
        with open(html_path, "w+", encoding='utf-8') as html_file:
            html = html_template.format(self._convert_global_style_to_css(), self.html)
            html_file.write(html)

    def _get_margin_vertical_and_horizontal(self):
        margin_vertical = 0
        margin_horizontal = 0
        if 'margin-left' in self.global_style['table']:
            margin_horizontal += int(self.global_style['table']['margin-left'].split("px")[0])
        elif 'margin' in self.global_style['table']:
            margin_horizontal += int(self.global_style['table']['margin'].split("px")[0]) / 2

        if 'margin-right' in self.global_style['table']:
            margin_horizontal += int(self.global_style['table']['margin-right'].split("px")[0])
        elif 'margin' in self.global_style['table']:
            margin_horizontal += int(self.global_style['table']['margin'].split("px")[0]) / 2

        if 'margin-top' in self.global_style['table']:
            margin_vertical += int(self.global_style['table']['margin-top'].split("px")[0])
        elif 'margin' in self.global_style['table']:
            margin_vertical += int(self.global_style['table']['margin'].split("px")[0]) / 2

        if 'margin-bottom' in self.global_style['table']:
            margin_vertical += int(self.global_style['table']['margin-bottom'].split("px")[0])
        elif 'margin' in self.global_style['table']:
            margin_vertical += int(self.global_style['table']['margin'].split("px")[0]) / 2
        return margin_horizontal, margin_vertical

    def _render_table_selenium(self, html_path, image_path, paper, driver):
        # driver = self.selenium_driver
        self._write_html_file(html_path)
        driver.get("file:///{}".format(os.path.abspath(html_path)))

        # required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        # required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        window_width = 6000
        window_height = 6000
        driver.set_window_size(window_width, window_height)

        # todo: 예외 처리 추가
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        image_sizes = []
        num_big_images = 0
        for img_element in img_elements:
            w = img_element.size['width']
            h = img_element.size['height']
            image_sizes.append([w, h])
            if w > self.meta['max_image_width'] or h > self.meta['max_image_height']:
                num_big_images += 1
                driver.execute_script("""
                var element = arguments[0];
                element.parentNode.removeChild(element);
                """, img_element)

        if num_big_images > 0:
            tds = driver.find_elements(By.TAG_NAME, 'td')
            num_td = len(tds)

            if num_big_images / num_td > self.meta['max_big_image_ratio']:
                return False

            if num_td <= self.meta['num_less_cell'] and num_big_images / num_td > self.meta[
                'max_big_image_ratio_when_less_cells']:
                return False

            if self.meta['max_empty_cell_ratio']:
                num_empty_tds = 0
                for td in tds:
                    if not td.text.strip():
                        num_empty_tds += 1
                if num_empty_tds / len(tds) > self.meta['max_empty_cell_ratio']:
                    return False

        table_element = driver.find_element(By.TAG_NAME, 'table')
        if num_big_images > 0:
            self.html = table_element.get_attribute('outerHTML')
        table_width = table_element.size['width']
        table_height = table_element.size['height']
        if not self.meta['table_full_size']:
            table_width = int(table_element.size['width'] * self.meta['relative_style']['table']['width_scale'])
            table_height = int(table_element.size['height'] * self.meta['relative_style']['table']['height_scale'])
        ar = table_width / table_height
        if self.meta['table_aspect_ratio'][0] > ar:
            table_height = int(table_width / self.meta['table_aspect_ratio'][0])
        elif self.meta['table_aspect_ratio'][1] < ar:
            table_width = int(table_height * self.meta['table_aspect_ratio'][1])
        # driver.close()
        # self.global_style['table']['width'] = str(table_width) + "px"
        # self.global_style['table']['height'] = str(table_height) + "px"
        #
        # self._write_html_file(html_path)
        # driver.get("file:///{}".format(os.path.abspath(html_path)))
        # driver.set_window_size(window_width, window_height)
        # table_element = driver.find_element(By.TAG_NAME, 'table')

        margin_horizontal, margin_vertical = self._get_margin_vertical_and_horizontal()
        image_width = table_width + margin_horizontal  # meta['margin_width']
        image_height = table_height + margin_vertical  # meta['margin_height']

        # self.global_style['table']['width'] = str(table_width) + "px"
        # self.global_style['table']['height'] = str(table_height) + "px"

        if paper is not None:
            paper_layer = paper.generate((image_width, image_height))
            base64_image = image_util.image_to_base64(paper_layer.image)
            self.global_style['#table_wrapper']['background-image'] = 'url("data:image/png;base64,{}")'.format(
                base64_image)
            self.global_style['#table_wrapper']['background-size'] = "100% 100%"

        self._write_html_file(html_path)

        driver.get("file:///{}".format(os.path.abspath(html_path)))
        driver.set_window_size(window_width, window_height)
        # driver.set_window_size(int(image_width * 1.5), int(image_height * 1.5))
        div_element = driver.find_element(By.ID, 'table_wrapper')
        div_element.screenshot(image_path)
        return True

    def effect(self, image):
        selectors = parse_config(self.meta["effect_config"])
        self.meta["distort"] = selectors["distort"].on()
        self.meta["rotate"] = selectors["rotate"].on()
        if not self.meta["distort"] and not self.meta["rotate"]:
            return image
        image = np.array(image)
        if self.meta["distort"]:
            effect_config = selectors["distort"].get().select()
            self.meta['table_effect'] = effect_config['name']
            if effect_config['name'] == "arc":
                arc = components.Arc(self.meta["effect_config"]["distort"]["arc"]["angles"],
                                     self.meta["effect_config"]["distort"]["arc"]["reverse"]["prob"])
                min_aspect_ratio = self.meta["effect_config"]["distort"]["arc"]["min_aspect_ratio"]
                height, width = image.shape[:2]
                if min_aspect_ratio <= width / height:
                    image = arc.apply_image(image)
            elif effect_config['name'] == "polynomial":
                polynomial = components.Polynomial(
                    self.meta["effect_config"]["distort"]["polynomial"]["dest_coord_ratios"],
                    self.meta["effect_config"]["distort"]["polynomial"]["move_prob"])
                image = polynomial.apply_image(image)
            elif effect_config['name'] == "sylinder":
                sylinder = components.Sylinder(self.meta["effect_config"]["distort"]["sylinder"]["angle"])
                image = sylinder.apply_image(image)
        if self.meta["rotate"]:
            rotate_config = selectors["rotate"].get()
            angle = rotate_config['angle'].select()
            ccw = rotate_config['ccw'].on()
            if not ccw:
                angle = -angle
            image = WandImage.from_array(image)
            image.rotate(angle)
            np.array(image)
        return image

    def render_table(self, image=None, paper=None, meta=None):
        self.meta = meta
        self.html = meta['html']

        image_path = None
        if not image:
            tmp_path = meta['tmp_path']
            image_path = os.path.join(tmp_path, str(uuid.uuid4()) + ".png")
            html_path = os.path.join(tmp_path, str(uuid.uuid4()) + ".html")

            try:
                self.global_style = meta['global_style']
                options = webdriver.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--detach_driver')
                driver = None
                driver = webdriver.Chrome('chromedriver', options=options)
                ret = self._render_table_selenium(html_path, image_path, paper, driver)
                if not ret:
                    return False
            except Exception as e:
                if os.path.isfile(image_path):
                    os.unlink(image_path)
                if os.path.isfile(html_path):
                    os.unlink(html_path)
                # raise e
                return False
            finally:
                if driver and hasattr(driver, "close"):
                    try:
                        driver.close()
                    except:
                        traceback.print_exc()
                        pass
                if driver and hasattr(driver, "quit"):
                    try:
                        driver.quit()
                    except:
                        traceback.print_exc()
                        pass

            image = Image.open(image_path)
            os.unlink(html_path)
            if '#table_wrapper' in self.global_style:
                if 'background-image' in self.global_style['#table_wrapper']:
                    del self.global_style['#table_wrapper']['background-image']
            meta['global_style'] = self.global_style
            self.meta['css'] = self._convert_global_style_to_css()

        image = self.effect(image)
        super().__init__(image)

        height, width = self.image.shape[:2]
        self.table_size = (width, height)

        if image_path:
            if hasattr(image, "close"):
                image.close()
            image = None
            os.unlink(image_path)

        return True
