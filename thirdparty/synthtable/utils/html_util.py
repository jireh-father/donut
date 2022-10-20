import re
from bs4 import BeautifulSoup

close_tag_regex = re.compile('</.*?>')
tag_regex = re.compile('<.*?>')
close_thead_regex = re.compile('</thead>')
thead_tbody_tag_regex = re.compile('(<tbody>|<thead>|</tbody>|</thead>)')


def remove_close_tags(text):
    """Remove html tags from a string"""

    return re.sub(close_tag_regex, '', text)


def remove_tags(text):
    """Remove html tags from a string"""

    return re.sub(tag_regex, '', text)


def insert_tbody_tag(html):
    if '<tbody>' not in html:
        if '<thead>' in html[:15]:
            html = close_thead_regex.sub('</thead><tbody>', html, 1)
        else:
            html = '<table><tbody>' + html[7:]
        html = html[:-8] + '</tbody></table>'
    return html


def remove_tag_in_table_cell(html, bs=None):
    if bs is None:
        bs = BeautifulSoup(html, 'html.parser')
    for td in bs.find_all("td"):
        content = "".join([str(tag) for tag in td.contents])
        td.string = remove_tags(content).strip()
    return str(bs)


def remove_thead_tbody_tag(html):
    return thead_tbody_tag_regex.sub("", html)
