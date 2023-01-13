# https://huggingface.co/course/chapter6/2?fw=pt
from transformers import AutoTokenizer, XLMRobertaTokenizer, MBartTokenizer
import argparse
import os
import glob
from donut import preprocess_label


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
            if args.use_image_tag:
                lines = f.readlines()
            else:
                lines = [preprocess_label(line.strip(), True) for line in f.readlines()]
            if not lines[-1].strip():
                del lines[-1]
            corpus_lines.extend(lines)
    # corpus_lines = open(args.corpus_path, encoding='utf-8').readlines()
    training_corpus = get_training_corpus(corpus_lines)

    old_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name)
    print(old_tokenizer)
    # old_xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-en")
    # print(old_xlmroberta_tokenizer)
    # old_mbart_tokenizer = MBartTokenizer.from_pretrained("hyunwoongko/asian-bart-en")
    # print(old_mbart_tokenizer)
    new_tokens = ['<tr>', '<td>'] + ['<td colspan="{}">'.format(i) for i in range(args.max_col_span)] + ['<td rowspan="{}">'.format(i) for i in range(args.max_row_span)]
    if args.use_thead:
        new_tokens += ['<thead>', '<tbody>']
    if args.use_image_tag:
        new_tokens += ['<img>']

    # new_tokens += ['&gt;', '&lt;']
    # old_tokenizer.add_tokens(new_tokens)
    # print("added tokens", old_tokenizer)
    print("training!")
    # print("old_tokenizer.vocab_size", old_tokenizer.vocab_size)

    if args.vocab_size:
        vocab_size = args.vocab_size
    else:
        vocab_size = old_tokenizer.vocab_size
        vocab_size += len(new_tokens)

    print("vocab_size", vocab_size)
    # tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, old_tokenizer.vocab_size + 22, new_special_tokens=new_tokens)  # 52000)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size,
                                                      new_special_tokens=new_tokens)  # 52000)
    print(tokenizer.encode("<tr><td>test<tr><td>"))
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    loaded_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    print(loaded_tokenizer)
    print(loaded_tokenizer.encode("<tr><td>test<tr><td>"))
    # xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained(args.output_dir)
    # print(xlmroberta_tokenizer)



    xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained(args.output_dir)
    if args.use_unk_token:
        total_tokens = set()
        for line in corpus_lines:
            total_tokens.update(set(line))
        try:
            total_tokens.remove(' ')
        except:
            pass
        unk_tokens = []
        for c in total_tokens:
            if 3 in xlmroberta_tokenizer.encode(c):
                unk_tokens.append(c)
        print(unk_tokens)
        print("add unk tokens", len(unk_tokens))
        xlmroberta_tokenizer.add_tokens(unk_tokens)
        # xlmroberta_tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(unk_tokens))})
        xlmroberta_tokenizer.save_pretrained(args.output_dir)

    print(xlmroberta_tokenizer)
    print(xlmroberta_tokenizer.encode("<tr><td>test<tr><td>"))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_paths', type=str,
                        default="D:\dataset\\table_ocr\crawling_train_corpus_without_except_chars\*_tokenizer_corpus.txt")
    parser.add_argument('--output_dir', type=str, default="D:\dataset/table_ocr/tokenizer_crawled_ko_no_imgtheadtag_span20_with_unk_tokens")
    parser.add_argument('--pretrained_name', type=str, default="hyunwoongko/asian-bart-ko")
    # parser.add_argument('--charset_path', type=str, default="thirdparty/synthtable/resources/charset/alphanum_special_korean.txt")

    parser.add_argument('--vocab_size', type=int, default=None)  # 100000)
    parser.add_argument('--max_row_span', type=int, default=20)  # 100000)
    parser.add_argument('--max_col_span', type=int, default=20)  # 100000)

    parser.add_argument('--use_thead', action='store_true', default=False)
    parser.add_argument('--use_image_tag', action='store_true', default=False)
    parser.add_argument('--use_unk_token', action='store_true', default=True)

    main(parser.parse_args())
