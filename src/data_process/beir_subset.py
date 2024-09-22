import random
import sys

sys.path.append('.')

import argparse
import json
import os.path

from src.data_process.util import chunk_corpus, generate_hash


def get_sampled_query_ids(qrels, sample):
    sampled_query_ids = None
    if sample is not None:
        sampled_query_ids = set()
        assert isinstance(sample, int)
        for item in qrels:
            query_id = item[0]
            sampled_query_ids.add(query_id)
        assert 0 < sample <= len(sampled_query_ids), f'sample size {sample} is invalid, check if it is in range (0, {len(sampled_query_ids)}]'
        sampled_query_ids = random.sample(list(sampled_query_ids), sample)
        print(f'{len(sampled_query_ids)} queries are sampled')
    return sampled_query_ids


def generate_dataset_with_relevant_corpus(split: str, qrels_path: str, chunk=False, num_query_sample=None, num_passage_sample=None):
    """

    @param split: split name, e.g., 'train', 'test'
    @param qrels_path: the path to BEIR subset's qrels file
    @return: None
    """
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]  # skip the first line
    print(f'#{split}', len(qrels))
    split_passage_hash_to_id = dict()
    split_query_ids = set()
    split_corpus = []  # output 1
    split_queries = []  # output 2
    query_to_corpus = {}  # output 3, query_id -> [corpus_id]

    sampled_query_ids = get_sampled_query_ids(qrels, num_query_sample)

    for idx, item in enumerate(qrels):  # for each line in qrels
        query_id = item[0]
        corpus_id = item[1]
        score = item[2]
        if int(score) == 0:
            continue
        if sampled_query_ids is not None and query_id not in sampled_query_ids:
            continue

        try:
            corpus_item = corpus[corpus_id]
        except KeyError:
            print(f'corpus_id {corpus_id} not found')
            continue
        query_item = queries[query_id]
        passage_content = corpus_item['title'] + '\n' + corpus_item['text']
        passage_hash = generate_hash(passage_content)

        if passage_hash not in split_passage_hash_to_id:  # make each passage unique
            split_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})
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
            corpus_item = corpus[c]
            passage_content = corpus_item['title'] + '\n' + corpus_item['text']
            passage_hash = generate_hash(passage_content)
            assert passage_hash in split_passage_hash_to_id, f'passage_hash {passage_hash} not found'
            passage_id = split_passage_hash_to_id[passage_hash]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': passage_id})

    # add sampled passages to corpus if num_passage_sample is larger than the num of collected passages
    if num_passage_sample is not None:
        assert 0 < num_passage_sample
        if len(split_corpus) < num_passage_sample:
            sampled_passage_ids = random.sample(list(corpus.keys()), num_passage_sample)
            sampled_passage_ids = set(sampled_passage_ids) # make each passage unique
            # add each passage to split_corpus if it is not already in split_corpus
            for c in sampled_passage_ids:
                corpus_item = corpus[c]
                passage_content = corpus_item['title'] + '\n' + corpus_item['text']
                passage_hash = generate_hash(passage_content)
                if passage_hash not in split_passage_hash_to_id:
                    split_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})
                    split_passage_hash_to_id[passage_hash] = corpus_item['_id']
                    if len(split_corpus) == num_passage_sample:
                        break

    if chunk:
        split_corpus = chunk_corpus(split_corpus)

    # save split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(split_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(split_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{chunk_state}.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


def generate_dataest_with_full_corpus(split, qrels_path: str, corpus_path: str, chunk=False, sample=None):
    chunk_state = '_chunk' if chunk else ''
    with open(qrels_path) as f:
        qrels = f.readlines()
    qrels = [q.split() for q in qrels[1:]]
    print(f'#{split}', len(qrels))
    split_query_ids = set()
    passage_hash_to_id = dict()  # To make unique corpus, use content hash rather than the original BEIR doc ID to avoid duplication
    full_corpus = []  # output 1
    split_queries = []  # output 2
    query_to_corpus = {}  # output 3, query_id -> [corpus_id]

    sampled_query_ids = get_sampled_query_ids(qrels, sample)

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
    corpus_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(full_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(full_corpus)}')

    # save split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{chunk_state}.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # save qrel json file processed from tsv file
    qrels_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='directory path to a BEIR subset')
    parser.add_argument('--corpus', type=str, choices=['full', 'relevant'], help='full or relevant corpus', default='full')
    parser.add_argument('-qs', '--query_sample', type=int)
    parser.add_argument('-ps', '--passage_sample', type=int)
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
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

    for split in ['train', 'dev', 'test']:
        if os.path.isfile(os.path.join(args.data, f'qrels/{split}.tsv')):
            if args.corpus == 'relevant':
                generate_dataset_with_relevant_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), args.chunk, args.query_sample, args.passage_sample)
            elif args.corpus == 'full':
                generate_dataest_with_full_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), os.path.join(args.data, 'corpus.json'), args.chunk, args.query_sample)
        else:
            print(f'{split} not found, skipped')
