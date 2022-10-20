"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from wand.image import Image
import uuid


class Arc(Component):
    def __init__(self, angles=None, reverse_prob=0.5):
        super().__init__()
        self.angles = angles
        self.reverse_prob = reverse_prob

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        angle = meta.get("angle", np.random.randint(self.angles[0], self.angles[1] + 1))

        meta = {
            "angle": angle,
            "reverse": np.random.rand() < self.reverse_prob
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        angle = meta["angle"]

        for layer in layers:
            im = Image.from_array(layer.image)
            im.virtual_pixel = 'transparent'
            im.distort('arc', (angle,))
            layer.image = np.array(im)

        return meta

    def apply_image(self, image):
        meta = self.sample(None)
        angle = meta["angle"]
        reverse = meta["reverse"]

        im = Image.from_array(image)
        im.virtual_pixel = 'transparent'
        if reverse:
            im.rotate(180)
        im.distort('arc', (angle,))
        if reverse:
            im.rotate(180)

        return np.array(im)
