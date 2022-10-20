import os

import numpy as np

from synthtiger import utils


def search_files(root, names=None, exts=None):
    paths = []

    for dir_path, _, file_names in os.walk(root, followlinks=True):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            file_ext = os.path.splitext(file_name)[1]

            if names is not None and file_name not in names:
                continue
            if exts is not None and file_ext.lower() not in exts:
                continue

            paths.append(file_path)

    return paths


class PathSelector:
    def __init__(self, paths=(), weights=(), exts=None, use_sort=True):
        super().__init__()
        self.paths = paths
        self.weights = weights if weights else [1] * len(paths)
        self.exts = exts
        self.use_sort = use_sort
        self._paths = []
        self._counts = []
        self._probs = np.array(self.weights) / sum(self.weights)
        self._update_paths()

    def select(self):
        return self._sample_path()

    def get(self, key, idx):
        return self._paths[key][idx]

    def get_path(self, key):
        return self.paths[key]

    def _update_paths(self):
        self._paths = []
        self._counts = []

        for path in self.paths:
            if not os.path.exists(path):
                continue

            paths = [path]
            if os.path.isdir(path):
                paths = search_files(path, exts=self.exts)
            if self.use_sort:
                paths.sort()
            self._paths.append(paths)
            self._counts.append(len(paths))

    def _sample_path(self):
        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no path: {self.paths[key]}")

        idx = np.random.randint(len(self._paths[key]))
        path = self._paths[key][idx]
        return path, key, idx
