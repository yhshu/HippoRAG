import argparse
import json

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/hotpotqa_corpus_legacy.json')
    parser.add_argument('--output', type=str, default='data/hotpotqa_corpus.json')
    args = parser.parse_args()

    old_corpus = json.load(open(args.input))
    new_corpus = []
    for idx, title in tqdm(enumerate(old_corpus)):
        text = ''.join(old_corpus[title])
        new_corpus.append({'idx': idx, 'title': title, 'text': text})

    json.dump(new_corpus, open(args.output, 'w'), indent=4)
    print(f'Converted {len(old_corpus)} passages to {len(new_corpus)} passages')
