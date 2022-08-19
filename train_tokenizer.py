# https://huggingface.co/course/chapter6/2?fw=pt
from transformers import AutoTokenizer
import argparse
import os


def get_training_corpus(corpus_lines):
    for start_idx in range(0, len(corpus_lines), 1000):
        samples = corpus_lines[start_idx: start_idx + 1000]
        yield samples


def main(args):
    corpus_lines = open(args.corpus_path, encoding='utf-8').readlines()

    training_corpus = get_training_corpus(corpus_lines)

    old_tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/asian-bart-en")
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, old_tokenizer.vocab_size + 1000)  # 52000)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    # tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")

    main(parser.parse_args())
