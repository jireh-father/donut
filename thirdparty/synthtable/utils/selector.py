"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from utils.switch import BoolSwitch


class Selector:
    def __init__(self, values, weights=None, postfix=None, prob=None):
        if isinstance(values, list):
            if len(values) == 2 and all(isinstance(c, float) for c in values):
                self.values = {'low': values[0], 'high': values[1]}
            elif len(values) == 2 and all(isinstance(c, int) for c in values):
                self.values = list(range(values[0], values[1] + 1))
            else:
                self.values = values
        else:
            self.values = [values]
        if weights is None:
            weights = [1] * len(self.values)
        self._probs = np.array(weights) / sum(weights)
        self.postfix = postfix
        if prob:
            self.bool_switch = BoolSwitch(prob)
        else:
            self.bool_switch = None

    def apply(self, layers, meta=None):
        assert isinstance(self.values, list)
        idx = np.random.choice(len(self.values), replace=False, p=self._probs)
        return self.values[idx].apply(layers, meta)

    def select(self):
        if self.bool_switch and not self.bool_switch.on():
            return None

        if isinstance(self.values, dict):
            return np.random.uniform(**self.values)
        else:
            idx = np.random.choice(len(self._probs), replace=False, p=self._probs)
            if self.postfix:
                return str(self.values[idx]) + self.postfix
            return self.values[idx]

    def on(self):
        if self.bool_switch:
            return self.bool_switch.on()
        return None



def parse_config(config):
    config_selector = {}
    for key in config:

        val = config[key]
        if isinstance(val, dict):
            if 'prob' in val:
                if 'values' in val:
                    config_selector[key] = Selector(**val)
                else:
                    if len(val) > 1:
                        config_selector[key] = BoolSwitch(prob=val['prob'], data=parse_config(
                            {sub_key: val[sub_key] for sub_key in val if sub_key != 'prob'}))
                    else:
                        config_selector[key] = BoolSwitch(prob=val['prob'])
            elif 'weight' in val:
                weights = []
                components = []
                for k in config:
                    v = config[k]
                    weights.append(v['weight'])
                    if len(v) > 1:
                        components.append({
                            'name': k,
                            'config': parse_config({sub_key: v[sub_key] for sub_key in v if sub_key != 'weight'})
                        })
                    else:
                        components.append({'name': k})

                return Selector(components, weights)
            elif 'values' in val:
                config_selector[key] = Selector(**val)
            else:
                config_selector[key] = parse_config(val)
        # elif isinstance(val, list):
        #     pass
        else:
            if not val:
                config_selector[key] = val
            else:
                config_selector[key] = Selector(val)
    return config_selector
