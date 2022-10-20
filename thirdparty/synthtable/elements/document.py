"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import components

from elements.content import Content


class Document:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.2])
        self.fullscreen = config.get("fullscreen", 0.5)
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [480, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.content = Content(config.get("content", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.ElasticDistortion()),
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(
                    components.Selector(
                        [
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                        ]
                    )
                )
            ],
            **config.get("effect", {}),
        )

    def generate(self, selenium_driver):
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        table_layer = self.content.generate(size, selenium_driver)#, int(self.short_size[1] * self.aspect_ratio[1]))

        fullscreen = np.random.rand() < self.fullscreen
        if fullscreen:
            bg_size = table_layer.table_size
        else:
            bg_width = table_layer.table_size[0] + (
                    table_layer.table_size[0] * np.random.uniform(self.margin[0], self.margin[1]))
            bg_height = table_layer.table_size[1] + (
                    table_layer.table_size[1] * np.random.uniform(self.margin[0], self.margin[1]))
            bg_size = (bg_width, bg_height)

        effect_meta = self.effect.apply([table_layer])
        table_layer.meta['document_effect'] = effect_meta

        return table_layer, bg_size
