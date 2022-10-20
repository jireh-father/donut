"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from wand.image import Image
import uuid


class Sylinder(Component):
    def __init__(self, angle=None):
        super().__init__()
        self.angle = angle

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        angle = meta.get("angle", np.random.randint(self.angle[0], self.angle[1] + 1))

        meta = {
            "angle": angle,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        angle = meta["angle"]

        for layer in layers:
            im = Image.from_array(layer.image)
            im.virtual_pixel = 'transparent'
            im.distort('plane_2_cylinder', (angle,))
            layer.image = np.array(im)

        return meta

    def apply_image(self, image):
        meta = self.sample(None)
        angle = meta["angle"]

        im = Image.from_array(image)
        im.virtual_pixel = 'transparent'
        im.distort('plane_2_cylinder', (angle,))

        return np.array(im)
