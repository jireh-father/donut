import re
from bs4 import BeautifulSoup
from html import unescape

close_tag_regex = re.compile('</.*?>')
tag_regex = re.compile('<.*?>')
close_thead_regex = re.compile('</thead>')
thead_tbody_tag_regex = re.compile('(<tbody>|<thead>|</tbody>|</thead>)')
multiple_space_regex = re.compile('\s+')
image_tag_regex = re.compile(r'\[\[\[img\]\]\]')
new_line_regex = re.compile("\n")
white_space_regex = re.compile('\s+')
space_tr_tag_regex = re.compile(r"\s?<tr>\s?")
space_td_tag_regex = re.compile(r"\s?<td>\s?")
space_img_tag_regex = re.compile(r"\s?<img>\s?")


def remove_multiple_spaces(text):
    return re.sub(multiple_space_regex, ' ', text)


def remove_close_tags(text):
    """Remove html tags from a string"""

    return re.sub(close_tag_regex, '', text)


def remove_tags(text):
    """Remove html tags from a string"""

    return re.sub(tag_regex, '', text)


def insert_tbody_tag(bs):
    tbody_tag = bs.new_tag("tbody")
    children = list(bs.table.children)
    bs.table.clear()
    bs.table.append(tbody_tag)
    for child in children:
        tbody_tag.append(child)
    return convert_bs_to_html_string(bs)


BLOCK_TAGS = [
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "textarea",
    # "li",
    "figcaption",
    "legend",
    "blockquote",
    "nav",
    "dt",
    "dd",
    "pre",
]

NEW_LINE_TAGS = [
    "br"
]

TABLE_TAGS = [
    "table",
    "thead",
    "tbody",
    "tr",
]


#
# A: 블록 태그안에 띄어쓰기 넣기 ( 해당되는 태그들 검색 )
# 1. 태그안에 첫, 끝에 띄어쓰기 넣기
# bs.find_all("li")[1].insert(0, " ")
# bs.find_all("li")[1].append(" ")
# 2. contents 돌면서 text일때만 앞뒤 띄어쓰기 넣기
# bs.find_all("li")[1].contents[0].replace_with("hoho")
# 3. br는 태그 자체를 " "로 변경
# bs.find_all("li")[1].contents[0].replace_with("hoho")

# B:
# 1. table, thead, tbody, tr attr 다 지우기 =>

# C: 태그 컨텐츠들 텍스트만 남기고 삭제(태그 삭제)
# 0. <img> 태그는 남겨두고 삭제되도록
# 1. <td> 돌면서 replace_with_childern
# 2. td에서 colspan, rowspan 만 빼고 attr 다 삭제

def set_li_marker(li_tag, ol_type, idx):
    if ol_type == "1":
        li_tag.replace_with(" {}. {} ".format(idx + 1, li_tag.text))
    elif ol_type == "a":
        li_tag.replace_with(" {}. {} ".format(chr(idx + 97), li_tag.text))
    elif ol_type == "A":
        li_tag.replace_with(" {}. {} ".format(chr(idx + 65), li_tag.text))
    elif ol_type == "i":
        li_tag.replace_with(" {}. {} ".format(chr(idx + 8560), li_tag.text))
    elif ol_type == "I":
        li_tag.replace_with(" {}. {} ".format(chr(idx + 8544), li_tag.text))


def _remove_tags(bs):
    for block_tag_name in BLOCK_TAGS:
        tags = bs.find_all(block_tag_name)
        if not tags:
            continue

        for tag in tags:
            tag.insert(0, " ")
            tag.append(" ")
            contents = tag.contents.copy()
            for child_tag in contents:
                if not child_tag.name:
                    child_tag.replace_with(" {} ".format(child_tag.text))

    for ol_tag in bs.find_all("ol"):
        ol_type = "1"
        if ol_tag.has_attr("type") and ol_tag["type"] in ["1", "a", "A", "i", "I"]:
            ol_type = ol_tag["type"]
        for idx, li_tag in enumerate(ol_tag.find_all("li").copy()):
            set_li_marker(li_tag, ol_type, idx)

    for ol_tag in bs.find_all("ul"):
        for idx, li_tag in enumerate(ol_tag.find_all("li").copy()):
            li_tag.replace_with(" {} {} ".format(chr(int('2022', 16)), li_tag.text))

    for new_line_tag_name in NEW_LINE_TAGS:
        tags = bs.find_all(new_line_tag_name)
        if not tags:
            continue

        for tag in tags:
            tag.replace_with(" ")

    for table_tag_name in TABLE_TAGS:
        tags = bs.find_all(table_tag_name)
        for tag in tags:
            tag.attrs = {}

    for td in bs.find_all("td"):
        img_tags = td.find_all("img")
        for img_tag in img_tags:
            img_tag.replace_with("[[[img]]]")
        if img_tags:
            text = re.sub(image_tag_regex, '<img>', td.text)
        else:
            text = td.text

        td.string = remove_multiple_spaces(text).strip()

        if td.attrs:
            keys = list(td.attrs.keys())
            for k in keys:
                if k not in ["colspan", "rowspan"]:
                    del td.attrs[k]
                else:
                    if td.attrs[k] == "1":
                        del td.attrs[k]


def remove_tag_in_table_cell(html, bs=None):
    if bs is None:
        bs = BeautifulSoup(html, 'html.parser')

    _remove_tags(bs)

    return convert_bs_to_html_string(bs)


def remove_thead_tbody_tag(html):
    return thead_tbody_tag_regex.sub("", html)


def convert_bs_to_html_string(bs):
    text = bs.text
    if "<" in text or ">" in text:
        html = str(bs)
        return unescape(html)
    else:
        return str(bs)


def remove_white_spaces(html):
    return white_space_regex.sub("", html)


def remove_new_line_and_multiple_spaces(html):
    html = re.sub(new_line_regex, " ", html)
    html = remove_multiple_spaces(html).strip()
    html = re.sub(space_tr_tag_regex, "<tr>", html)
    html = re.sub(space_td_tag_regex, "<td>", html)
    html = re.sub(space_img_tag_regex, "<img>", html)
    return html
