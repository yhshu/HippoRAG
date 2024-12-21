import argparse
import json
import os
import pickle

from tqdm import tqdm

from src.data_process.util import generate_hash
from src.langchain_util import num_tokens_by_tiktoken


def read_enwiki_corpus():
    full_corpus = []
    contents = []
    full_corpus_path = 'data/enwiki_full_corpus.pkl'
    contents_path = 'data/enwiki_contents.pkl'

    if os.path.exists(full_corpus_path) and os.path.exists(contents_path):
        with open(full_corpus_path, 'rb') as f:
            full_corpus = pickle.load(f)
        with open(contents_path, 'rb') as f:
            contents = pickle.load(f)
        return full_corpus, contents
    else:
        content_hash_set = set()
        with open(args.wiki) as f:
            for line in tqdm(f, 'Processing wiki text'):
                item = json.loads(line.strip())
                title = item['title'] + ' - ' + item['section']
                text = item['text']
                content = title + '\n' + text
                content_hash = generate_hash(content)
                if content_hash not in content_hash_set:
                    content_hash_set.add(content_hash)
                    full_corpus.append({'idx': len(full_corpus), 'title': title, 'text': text})
                    contents.append(content)
        with open(full_corpus_path, 'wb') as f:
            pickle.dump(full_corpus, f)
        with open(contents_path, 'wb') as f:
            pickle.dump(contents, f)
        return full_corpus, contents

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki', type=str, default='/fs/project/PAS1576/yiheng/workspace/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl')
    args = parser.parse_args()

    full_corpus, contents = read_enwiki_corpus()
    num_tokens = 0

    with tqdm(enumerate(contents), total=len(contents), desc='Counting tokens', ncols=100) as t:
        for idx, c in t:
            num_tokens += num_tokens_by_tiktoken(c)
            avg_tokens = num_tokens / (idx + 1)
            t.set_postfix({'Avg tokens per document': avg_tokens})