"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from wand.image import Image
from collections import namedtuple


class Polynomial(Component):
    def __init__(self, dest_coord_ratios=None, prob=0.5):
        super().__init__()
        self.dest_coord_ratios = dest_coord_ratios
        self.Point = namedtuple('Point', ['x', 'y', 'i', 'j'])
        self.prob = prob

    def _sample_dest_coord(self, is_start, length):
        if np.random.rand() < self.prob:
            if is_start:
                return int(length * np.random.uniform(self.dest_coord_ratios[0], self.dest_coord_ratios[1]))
            else:
                return int(length - length * np.random.uniform(self.dest_coord_ratios[0], self.dest_coord_ratios[1]))
        else:
            if is_start:
                return 0
            else:
                return length

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        return meta

    def _sample(self, meta=None):

        if meta is None:
            meta = {}
        width = meta['width']
        height = meta['height']

        alpha = self.Point(0, 0, self._sample_dest_coord(True, width), self._sample_dest_coord(True, height))
        beta = self.Point(width, 0, self._sample_dest_coord(False, width), self._sample_dest_coord(True, height))
        gamma = self.Point(width, height, self._sample_dest_coord(False, width), self._sample_dest_coord(False, height))
        delta = self.Point(0, height, self._sample_dest_coord(True, width), self._sample_dest_coord(False, height))

        args = (
            1.5, alpha.x, alpha.y, alpha.i, alpha.j, beta.x, beta.y, beta.i, beta.j, gamma.x, gamma.y, gamma.i, gamma.j,
            delta.x, delta.y, delta.i, delta.j,)

        meta = {
            "polynomial_args": args
        }

        return meta

    def apply(self, layers, meta=None):

        for layer in layers:
            im = Image.from_array(layer.image)
            meta['width'] = im.width
            meta['height'] = im.height
            meta = self._sample(meta)
            polynomial_args = meta["polynomial_args"]
            im.virtual_pixel = 'transparent'
            im.distort('polynomial', polynomial_args)
            layer.image = np.array(im)

        return meta

    def apply_image(self, image):
        meta = {}
        im = Image.from_array(image)
        meta['width'] = im.width
        meta['height'] = im.height
        meta = self._sample(meta)
        polynomial_args = meta["polynomial_args"]
        im.virtual_pixel = 'transparent'
        im.distort('polynomial', polynomial_args)
        return np.array(im)
