import sys

sys.path.append('.')

import random
from gritlm import GritLM
from tqdm import tqdm
from src.processing import corpus_has_duplication

import argparse
import json
import os.path

from src.data_process.util import chunk_corpus, generate_hash


def get_sampled_query_ids(qrels, num_sample):
    sampled_query_ids = None
    if num_sample is not None:
        sampled_query_ids = set()
        for item in qrels:
            query_id = item[0]
            sampled_query_ids.add(query_id)

        if num_sample.isdigit() or isinstance(num_sample, int):
            num_sample = int(num_sample)
            assert 0 < num_sample <= len(sampled_query_ids), f'sample size {num_sample} is invalid, check if it is in range (0, {len(sampled_query_ids)}]'
            sampled_query_ids = random.sample(list(sampled_query_ids), num_sample)
            print(f'{len(sampled_query_ids)} queries are sampled')
        elif num_sample == 'all':
            print('All queries are sampled')
        else:
            raise ValueError(f'Invalid sample size: {num_sample}')
    return sampled_query_ids


def generate_dataset_with_relevant_corpus(split: str, qrels_path: str, full_corpus, chunk=False, num_query=None, passage_per_query=None, retriever_name='bm25', dataset_name=None):
    """

    @param split: split name, e.g., 'train', 'test'
    @param qrels_path: the path to BEIR subset's qrels file
    @return: None
    """
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]  # skip the first line
    print(f'#{split} qrels', len(qrels))
    split_passage_hash_to_id = dict()
    split_query_ids = set()
    full_corpus_ids = set()  # all passage ids in qrels
    split_corpus = []  # output 1
    split_queries = []  # output 2
    query_to_corpus = {}  # output 3, query_id -> [corpus_id]

    sampled_query_ids = get_sampled_query_ids(qrels, num_query)
    sample_str = f'_{len(sampled_query_ids)}' if sampled_query_ids is not None else ''

    for idx, item in enumerate(qrels):  # for each line in qrels
        query_id = item[0]
        corpus_id = item[1]
        full_corpus_ids.add(corpus_id)
        score = item[2]
        if int(score) == 0:
            continue
        if sampled_query_ids is not None and query_id not in sampled_query_ids:
            continue

        try:
            corpus_item = full_corpus[corpus_id]
        except KeyError:
            print(f'corpus_id {corpus_id} not found')
            continue
        query_item = queries[query_id]
        passage_content = corpus_item['title'] + '\n' + corpus_item['text']
        passage_hash = generate_hash(passage_content)

        if passage_hash not in split_passage_hash_to_id:  # make each passage unique
            split_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})  # positive passage
            split_passage_hash_to_id[passage_hash] = corpus_item['_id']
        if query_item['_id'] not in split_query_ids:  # make each query unique
            split_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
            split_query_ids.add(query_item['_id'])
        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = {}
        query_to_corpus[query_id][corpus_id] = int(score)

    # add supporting passages to query info
    for query in split_queries:
        query['paragraphs'] = []
        for c in query_to_corpus[query['_id']]:
            corpus_item = full_corpus[c]
            passage_content = corpus_item['title'] + '\n' + corpus_item['text']
            passage_hash = generate_hash(passage_content)
            assert passage_hash in split_passage_hash_to_id, f'passage_hash {passage_hash} not found'
            passage_id = split_passage_hash_to_id[passage_hash]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': passage_id})

    assert corpus_has_duplication(split_corpus) is False, "Duplicated passages found in split_corpus"

    # add sampled passages to corpus if num_passage_sample is larger than the num of collected passages
    if passage_per_query is not None:
        assert 0 < passage_per_query <= len(full_corpus_ids), f'passage_per_query {passage_per_query} is invalid, check if it is in range (0, {len(full_corpus_ids)})'
        full_corpus_values = list(full_corpus.values())
        # sampled_corpus_values = random.sample(full_corpus_values, min(len(full_corpus_values), 100 * len(split_queries)))
        corpus_contents = [item['title'] + '\n' + item['text'] for item in full_corpus_values]
        print(f'#{split} retrieving from corpus', len(corpus_contents))

        if retriever_name == 'bm25':
            from src.pangu.retrieval_api import BM25SparseRetriever
            retriever = BM25SparseRetriever(corpus_contents, f"{dataset_name}_{split}_{len(corpus_contents)}")
        elif 'GritLM' in retriever_name:
            from src.pangu.retrieval_api import GritLMRetriever
            gritlm_model = GritLM(retriever_name, torch_dtype='auto')
            retriever = GritLMRetriever(corpus_contents, retriever_name, gritlm_model)

        for query in tqdm(split_queries, desc='Sampling distractors'):
            if len(query['paragraphs']) >= passage_per_query:
                continue
            query_text = query['text']
            top_k_indices = retriever.get_top_k_indices(query_text, passage_per_query + 50, distinct=True)
            num_distractor = 0
            for i in top_k_indices:
                passage_content = full_corpus_values[i]['title'] + '\n' + full_corpus_values[i]['text']
                passage_hash = generate_hash(passage_content)
                if passage_hash not in split_passage_hash_to_id:
                    # add to split_corpus as a distractor
                    split_corpus.append({'title': full_corpus_values[i]['title'], 'text': full_corpus_values[i]['text'], 'idx': full_corpus_values[i]['_id']})
                    split_passage_hash_to_id[passage_hash] = full_corpus_values[i]['_id']
                    num_distractor += 1
                if num_distractor == passage_per_query - len(query['paragraphs']):
                    break
            assert corpus_has_duplication(split_corpus) is False, "Duplicated passages found in split_corpus"

    if chunk:
        split_corpus = chunk_corpus(split_corpus)

    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(split_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(split_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


def generate_dataest_with_full_corpus(split: str, qrels_path: str, corpus_path: str, chunk: bool = False, sample: int = None):
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]
    print(f'#{split} qrels', len(qrels))
    split_query_ids = set()
    passage_hash_to_id = dict()  # To make unique corpus, use content hash rather than the original BEIR doc ID to avoid duplication
    full_corpus = []  # output 1
    split_queries = []  # output 2
    query_to_corpus = {}  # output 3, query_id -> [corpus_id]

    sampled_query_ids = get_sampled_query_ids(qrels, sample)
    sample_str = f'_{len(sampled_query_ids)}' if sampled_query_ids is not None else ''

    # process qrels
    for idx, item in enumerate(qrels):
        query_id = item[0]
        corpus_id = item[1]
        score = item[2]
        if int(score) == 0:
            continue
        if sampled_query_ids is not None and query_id not in sampled_query_ids:
            continue
        query_item = queries[query_id]

        if query_item['_id'] not in split_query_ids:  # make each query unique
            split_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
            split_query_ids.add(query_item['_id'])
        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = {}
        query_to_corpus[query_id][corpus_id] = int(score)

    # read jsonl file to get full corpus
    with open(corpus_path) as f:
        # read each line as json
        for line in f:
            item = json.loads(line)
            passage_content = item['title'] + '\n' + item['text']
            passage_hash = generate_hash(passage_content)
            if passage_hash not in passage_hash_to_id:
                full_corpus.append({'title': item['title'], 'text': item['text'], 'idx': item['_id']})
                passage_hash_to_id[passage_hash] = item['_id']
        print(f'#{split} corpus', len(full_corpus))

    # add supporting passages to query info
    for query in split_queries:
        query['paragraphs'] = []
        for c in query_to_corpus[query['_id']]:  # for each relevant passage
            if c not in corpus:
                print(f'corpus_id {c} not found')
                continue
            corpus_item = corpus[c]
            passage_content = corpus_item['title'] + '\n' + corpus_item['text']
            passage_hash = generate_hash(passage_content)
            assert passage_hash in passage_hash_to_id, f'passage_hash {passage_hash} not found'
            passage_id = passage_hash_to_id[passage_hash]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': passage_id})

    if chunk:
        full_corpus = chunk_corpus(full_corpus)
    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(full_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(full_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{sample_str}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='directory path to a BEIR subset')
    parser.add_argument('--corpus', type=str, choices=['full', 'relevant'], help='full or relevant corpus', default='full')
    parser.add_argument('-ntrain', '--num_train', help='number of training samples')
    parser.add_argument('-ndev', '--num_dev', help='number of dev samples')
    parser.add_argument('-ntest', '--num_test', help='number of test samples')
    parser.add_argument('-pq', '--passage_per_query', type=int, help='number of passages per query to sample, only used when corpus is `relevant`')
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--retriever', type=str, default='bm25', help='retrieve for sampling distractors')
    parser.add_argument('--dataset', type=str, help='dataset name')
    args = parser.parse_args()

    random.seed(args.seed)

    print(args)
    subset_name = args.data.split('/')[-1]  # BEIR subset name
    with open(os.path.join(args.data, 'queries.jsonl')) as f:
        queries = f.readlines()
    queries = [json.loads(q) for q in queries]  # jsonl -> list of json
    queries = {q['_id']: q for q in queries}  # dict: query_id -> query

    with open(os.path.join(args.data, 'corpus.jsonl')) as f:
        corpus = f.readlines()
    corpus = [json.loads(c) for c in corpus]
    corpus = {c['_id']: c for c in corpus}

    split_num_sample = {'train': args.num_train, 'dev': args.num_dev, 'test': args.num_test}

    for split in split_num_sample:
        num_sample = split_num_sample[split]
        if num_sample is None:
            continue
        if os.path.isfile(os.path.join(args.data, f'qrels/{split}.tsv')):
            if args.corpus == 'relevant':
                generate_dataset_with_relevant_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), corpus, args.chunk, num_sample,
                                                      args.passage_per_query, args.retriever, args.dataset)
            elif args.corpus == 'full':
                generate_dataest_with_full_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), os.path.join(args.data, 'corpus.json'),
                                                  args.chunk, num_sample)
        else:
            print(f'{split} not found, skipped')
