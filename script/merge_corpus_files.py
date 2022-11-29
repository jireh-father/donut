# https://huggingface.co/course/chapter6/2?fw=pt
from transformers import AutoTokenizer, XLMRobertaTokenizer, MBartTokenizer
import argparse
import os
import glob


def get_training_corpus(corpus_lines):
    for start_idx in range(0, len(corpus_lines), 1000):
        samples = corpus_lines[start_idx: start_idx + 1000]
        yield samples


def main(args):
    corpus_paths = glob.glob(args.corpus_paths)
    if not corpus_paths:
        corpus_paths = args.corpus_paths.split(",")
    corpus_lines = []
    for corpus_path in corpus_paths:
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines[-1].strip():
                del lines[-1]
            corpus_lines.extend(lines)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("".join(corpus_lines))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_paths', type=str,
                        default="D:\dataset\\table_ocr\crawling_train_corpus\*_cell_text_corpus.txt")
    parser.add_argument('--output_path', type=str, default="D:\dataset/table_ocr/crawling_train_corpus/total_cell_text_corpus.txt")

    main(parser.parse_args())
