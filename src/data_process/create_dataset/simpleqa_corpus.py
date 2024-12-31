import argparse
import json
import os
import random

from tqdm import tqdm

from src.data_process.create_dataset.wiki_corpus import read_enwiki_corpus
from src.data_process.util import generate_hash
from src.pangu.retrieval_api import BM25SparseRetriever

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki', type=str, default='/fs/project/PAS1576/yiheng/workspace/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl')
    args = parser.parse_args()

    dataset = json.load(open('data/simpleqa.json'))

    random.seed(1)

    full_corpus, contents = read_enwiki_corpus()

    os.makedirs('data/bm25_sparse/wiki_text', exist_ok=True)
    bm25_retriever = BM25SparseRetriever(contents, 'data/bm25_sparse/wiki_text')

    k_list = [2, 6, 20, 80]
    max_k = max(k_list)

    corpora = {k: [] for k in k_list}
    hash_sets = {k: set() for k in k_list}

    for sample in tqdm(dataset, desc='Collecting relevant wiki text'):
        question_top_indices = bm25_retriever.get_top_k_indices(sample['question'], max_k, True, False)
        answer_top_indices = bm25_retriever.get_top_k_indices(sample['answer'], max_k, True, False)

        for k in k_list:
            question_k = question_top_indices[:k]
            answer_k = answer_top_indices[:k]

            combined_indices = []
            seen = set()
            for idx in question_k + answer_k:
                if idx not in seen:
                    seen.add(idx)
                    combined_indices.append(idx)

            for idx in combined_indices:
                content = full_corpus[idx]['text'].strip()
                content_hash = generate_hash(content)
                if content_hash not in hash_sets[k]:
                    hash_sets[k].add(content_hash)
                    corpora[k].append(full_corpus[idx])

    for k in k_list:
        corpus = corpora[k]
        corpus_filename = f'data/simpleqa_{len(corpus)}_k{k}_corpus.json'
        with open(corpus_filename, 'w') as f:
            json.dump(corpus, f)
            print(f'Saving {len(corpus)} passages to {corpus_filename}')

        queries_filename = f'data/simpleqa_{len(corpus)}_k{k}.json'
        with open(queries_filename, 'w') as f:
            json.dump(dataset, f)
            print(f'Saving {len(dataset)} queries to {queries_filename}')
