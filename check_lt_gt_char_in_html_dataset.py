# https://huggingface.co/course/chapter6/2?fw=pt
import argparse
import json


def main(args):
    results = []
    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        item = json.loads(line)
        for tokens in item['html']['cells']:
            text = "".join(tokens['tokens'])
            if "&lt;" in text or "&gt;" in text:
                print(text)
                results.append(text)
    print(results)
    print(len(results))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")

    main(parser.parse_args())
