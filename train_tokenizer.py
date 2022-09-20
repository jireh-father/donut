# https://huggingface.co/course/chapter6/2?fw=pt
from transformers import AutoTokenizer, XLMRobertaTokenizer, MBartTokenizer
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
    print(old_tokenizer)
    # old_xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-en")
    # print(old_xlmroberta_tokenizer)
    # old_mbart_tokenizer = MBartTokenizer.from_pretrained("hyunwoongko/asian-bart-en")
    # print(old_mbart_tokenizer)
    new_tokens = ['<tr>', '<td>'] + ['<tdcolspan="{}">'.format(i) for i in range(10)] + ['<tdrowspan="{}">'.format(i) for i in range(10)]
    if args.use_thead:
        new_tokens += ['<thead>', '<tbody>']
    # new_tokens += ['&gt;', '&lt;']
    # old_tokenizer.add_tokens(new_tokens)
    # print("added tokens", old_tokenizer)
    print("training!")
    # print("old_tokenizer.vocab_size", old_tokenizer.vocab_size)

    if args.vocab_size:
        vocab_size = args.vocab_size
    else:
        if args.use_thead:
            vocab_size = old_tokenizer.vocab_size + 24
        else:
            vocab_size = old_tokenizer.vocab_size + 26
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

    mbart_tokenizer = MBartTokenizer.from_pretrained(args.output_dir)

    print(mbart_tokenizer)
    print(mbart_tokenizer.encode("<tr><td>test<tr><td>"))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet/train_corpus.txt")
    parser.add_argument('--output_dir', type=str, default="D:\dataset/table_ocr/pubtabnet/tokenizer_use_head_train_corpus")
    parser.add_argument('--vocab_size', type=int, default=None)#100000)
    parser.add_argument('--use_thead', action='store_true', default=True)

    main(parser.parse_args())
