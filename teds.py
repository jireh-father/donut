# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm
from .parallel import parallel_process
import re


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


transtab = str.maketrans({key: None for key in r"""!#$%&'()*+,-.:;?@[\]^_`{|}~"""})
transtab_post = str.maketrans({key: None for key in r"""<>"=/"""})
tag_special_re = re.compile(r'[<>"=/]')


def _filter_token(token, decode_td_span_tag=False):
    if token.startswith("<td") or token in ['<tr>', '</td>', '</tr>']:
        if decode_td_span_tag and (token.startswith("<tdr") or token.startswith("<tdc")):
            return "<td {}".format(token[3:])
        else:
            return token
    else:
        return token.translate(transtab_post).strip()


def preprocess_tag_str(tag_str, decode_td_span_tag=False, remove_close_tag=False, remove_content_token=False):
    caption = tag_str.translate(transtab).strip()
    caption_token_list = []
    for token in caption.strip().split():
        if remove_close_tag and token.startswith("</t"):
            continue
        if remove_content_token and not token.startswith("<t") and not token.startswith("</t"):
            continue
        caption_token_list.append(_filter_token(token, decode_td_span_tag))

    return caption_token_list


def decode_to_html(tags, restore_close_tag=False):
    html_str = ""
    for i, tag in enumerate(tags):
        if restore_close_tag and tag == "<tr>" and i > 0:
            html_str += "</td></tr>"
        if restore_close_tag and tag == "<td>" and tags[i - 1] != "<tr>":
            html_str += "</td>"

        if tag.startswith("<"):
            html_str += tag
        else:
            if i == 0 or tags[i - 1].startswith("<"):
                html_str += tag
            else:
                html_str += " {}".format(tag)
    if restore_close_tag:
        html_str += "</td></tr>"
    return html_str


def postprocess_html_tag(html_tag):
    html_tag = html_tag.replace("<td", "</td><td")
    html_tag = html_tag.replace("<tr></td>", "<tr>")
    html_tag = html_tag.replace("<tdrowspan", "<td rowspan")
    html_tag = html_tag.replace("<tdcolspan", "<td colspan")
    html_tag = html_tag.replace("<tr>", "</td></tr><tr>")
    html_tag = html_tag[10:]
    html_tag += "</td></tr>"
    html_tag = html_tag.replace("<td> ", "<td>")
    return "<table>{}</table>".format(html_tag.replace(" </td>", "</td>"))


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        try:
            pred = html.fromstring(pred, parser=parser)
            true = html.fromstring(true, parser=parser)
        except:
            return 0.0
        n_nodes_pred = len(pred.xpath(".//*"))
        n_nodes_true = len(true.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)
        tree_pred = self.load_html_tree(pred)
        tree_true = self.load_html_tree(true)
        print(tree_pred)
        print(tree_true)
        distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        return 1.0 - (float(distance) / n_nodes)

    def batch(self, pred_htmls, true_htmls):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_htmls[i], true_htmls[i]) for i in tqdm(range(len(true_htmls)))]
        else:
            inputs = [{'pred': pred_htmls[i], 'true': true_htmls[i]} for i in range(len(true_htmls))]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        return scores
