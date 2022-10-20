import numpy as np


class Switch:
    def __init__(self, component, prob=1):
        self.component = component
        self.prob = prob

    def on(self, layers, meta=None):
        if np.random.rand() < self.prob:
            return self.component.apply(layers, meta)
        return None


class BoolSwitch:
    def __init__(self, prob=1, data=None):
        self.prob = prob
        self.data = data

    def on(self):
        return np.random.rand() < self.prob

    def get(self):
        return self.data
