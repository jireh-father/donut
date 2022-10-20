"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import traceback

import components as comps
from layers import *
from utils.selector import Selector, parse_config


class Table():
    def __init__(self, config):
        self.table_selector = parse_config(config)

        for selector_dict in self.table_selector.values:
            name = selector_dict['name']
            if name == 'static':
                self.static = comps.StaticTable(selector_dict['config'], config["static"])
            else:
                self.synth = comps.SynthTable(selector_dict['config'], config["synth"])

    def generate(self, size, selenium_driver):#, max_size):
        table_layer = TableLayer(size, selenium_driver)

        table_config = self.table_selector.select()
        # try:
        table_type = table_config['name']
        if table_type == "static":
            self.static.apply([table_layer], {'size': size})
        elif table_type == "synth":
            self.synth.apply([table_layer])
        table_layer.meta['table_type'] = table_type
        # except:
        #     traceback.print_exc()
        #     return self.generate(size, selenium_driver)#, max_size)

        return table_layer
