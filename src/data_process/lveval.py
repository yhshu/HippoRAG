import argparse
import copy
import json

from tqdm import tqdm

from src.data_process.util import generate_hash
from src.pangu.retrieval_api import BM25SparseRetriever

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/LVEval/hotpotwikiqa_mixup/hotpotwikiqa_mixup_256k.jsonl')
    args = parser.parse_args()

    # read jsonl from LM Eval
    with open(args.data, 'r') as f:
        data = [json.loads(line) for line in f]
    print('#query', len(data))

    corpus = []
    corpus_hash_set = set()
    dataset = []

    for sample in tqdm(data, desc='Collecting data'):
        query = sample['input']
        context = sample['context']

        passage_split = context.split('### Passage ')
        passages = []
        for p in passage_split:
            if p.strip() == '':
                continue
            p = '\n'.join(p.split('\n')[1:])
            passages.append(p)

        bm25_retriever = BM25SparseRetriever(passages)
        selected_passages = bm25_retriever.get_top_k_sentences(query, 200, True)
        for passage in selected_passages:
            passage_hash = generate_hash(passage)
            if passage_hash in corpus_hash_set:
                continue
            corpus_hash_set.add(passage_hash)
            corpus.append({'idx': len(corpus), 'title': '', 'text': passage})

        new_sample = copy.deepcopy(sample)
        new_sample['question'] = query
        del new_sample['input']
        dataset.append(new_sample)

    dataset_output_path = 'data/lveval.json'
    with open(dataset_output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f'Dataset {len(dataset)} write to', dataset_output_path)

    corpus_output_path = 'data/lveval_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f, indent=4)
    print(f'Corpus {len(corpus)} write to', corpus_output_path)
