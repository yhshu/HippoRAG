import sys

sys.path.append('.')

from src.processing import query_data_has_duplication, corpus_has_duplication
import json
import os
import random

import pandas as pd
from tqdm import tqdm

from src.data_process.util import generate_hash
from src.pangu.retrieval_api import BM25SparseRetriever

if __name__ == '__main__':
    df = pd.read_csv('data/popQA.tsv', sep='\t')

    data = []
    for row in df.iterrows():
        data.append(json.loads(row[1].to_json()))

    random.seed(1)
    data = random.sample(data, 1000)

    with open('data/popqa.json', 'w') as f:
        json.dump(data, f)
        print(f'{len(data)} samples saved to data/popqa.json')

    full_corpus = []
    contents = []
    content_hash_set = set()
    with open('/research/nfs_su_809/workspace/shu.251/atlas/corpus/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl') as f:
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

    os.makedirs('data/bm25_sparse/wiki_text', exist_ok=True)
    bm25_retriever = BM25SparseRetriever(contents, 'data/bm25_sparse/wiki_text')

    corpus = []
    corpus_content_hash_set = set()
    for sample in tqdm(data, 'Collecting relevant wiki text'):
        k = 10
        top_indices = bm25_retriever.get_top_k_indices(sample['question'], k, True, False)
        assert len(top_indices) == k
        for idx in top_indices:
            content = full_corpus[idx]['title'] + '\n' + full_corpus[idx]['text']
            content_hash = generate_hash(content)
            if content_hash not in corpus_content_hash_set:
                corpus_content_hash_set.add(content_hash)
                corpus.append(full_corpus[idx])

    with open('data/popqa_corpus.json', 'w') as f:
        json.dump(corpus, f)
        print(f'{len(corpus)} passages saved to data/popqa_corpus.json')

    query_data_has_duplication(data, False, False)
    corpus_has_duplication(corpus)
