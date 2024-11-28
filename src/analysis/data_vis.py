import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    data = json.load(open(f"data/{args.dataset}.json", 'r'))
    print(len(data))
    corpus = json.load(open(f"data/{args.dataset}_corpus.json", 'r'))
    print(len(corpus))
