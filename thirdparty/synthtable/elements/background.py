"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from synthtiger import components, layers
from utils.selector import Selector


class Background:
    def __init__(self, config):
        self.image = components.BaseTexture(**config.get("image", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.bg_type_selector = Selector(["image", "color"], [config["weights"]["image"], config["weights"]["color"]])
        self.effect = components.Iterator(
            [
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {})
        )

    def generate(self, size):
        bg_layer = layers.RectLayer(size, (255, 255, 255, 255))
        if self.bg_type_selector.select() == "image":
            bg_image_meta = self.image.apply([bg_layer])
        else:
            bg_image_meta = self.color.apply([bg_layer])

        bg_effect_meta = self.effect.apply([bg_layer])

        return bg_layer, bg_image_meta, bg_effect_meta
