import argparse
import json

from tqdm import tqdm

from src.langchain_util import num_tokens_by_tiktoken

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/hotpotqa_corpus_legacy.json')
    parser.add_argument('--output', type=str, default='data/hotpotqa_corpus.json')
    args = parser.parse_args()

    old_corpus = json.load(open(args.input))
    new_corpus = []

    num_token = 0
    for idx, title in tqdm(enumerate(old_corpus)):
        text = ''.join(old_corpus[title])
        num_token += num_tokens_by_tiktoken(title) + num_tokens_by_tiktoken(text)
        new_corpus.append({'idx': idx, 'title': title, 'text': text})
    print(f'Average number of tokens: {num_token / len(old_corpus)}')

    json.dump(new_corpus, open(args.output, 'w'), indent=4)
    print(f'Converted {len(old_corpus)} passages to {len(new_corpus)} passages')
