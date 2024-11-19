import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    corpus = json.load(open(f"data/{args.dataset}_corpus.json"))
    new_corpus = []
    for c in corpus:
        if 'idx' in c:
            del c['idx']
        new_corpus.append(c)
    with open(f"data/{args.dataset}_corpus.json", "w") as f:
        json.dump(new_corpus, f)
