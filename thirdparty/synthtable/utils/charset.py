"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger import utils


class Charset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self._charset = set()
        self._update_charset()

    def _update_charset(self):
        self._charset = utils.read_charset(self.path)

    def check_charset(self, text):
        if len(set(text) - self._charset) > 0:
            return False
        return True
