import argparse


def _pad_text_side_with_spaces(tags):
    for tag in tags:
        if tag.name:
            if hasattr(tag, "contents"):
                _pad_text_side_with_spaces(tag.contents)
        else:
            tag.replace_with(" {} ".format(tag.text))


def main(args):
    char_thr = int('e000', 16)

    corpus_paths = args.corpus_paths.split(",")
    for corpus_path in corpus_paths:
        for i, line in enumerate(open(corpus_path, 'r', encoding='utf-8')):
            if i % 1000 == 0:
                print(i)
            line = line.strip()
            if not line:
                continue

            for c in line:
                if ord(c) >= char_thr:
                    print(c, ord(c), hex(ord(c)), line, corpus_path)
                    break

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_paths', type=str,
                        default="D:\dataset\\table_ocr\crawling_train_corpus_without_except_chars_and_img_tags\\total_cell_text_corpus.txt")
    main(parser.parse_args())
