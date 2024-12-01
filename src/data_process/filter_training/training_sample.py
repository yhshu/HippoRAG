import sys

sys.path.append('.')

import argparse
import json
import os
import random

from tqdm import tqdm
from src.hipporag import HippoRAG
from src.ircot_hipporag import get_gold_docs, get_oracle_triples


def collect_filter_data(dataset_name: str, num_sample: int, num_before_filter: int = 5, extractor='gpt-4o-mini',
                        graph_creating_retriever='GritLM/GritLM-7B', linker='GritLM/GritLM-7B'):
    res = []
    input_path = f'data/{dataset_name}.json'
    data = json.load(open(input_path))
    print(f'Loaded {len(data)} samples from {input_path}')
    data = random.sample(data, min(num_sample, len(data)))

    hipporag = HippoRAG(dataset_name, 'openai', extractor, graph_creating_retriever, 'ner',
                        'facts_and_sim_passage_node_unidirectional', 0.8, True, False,
                        None, False, 'ppr', 0.5,0.9, None,
                        None, linker, None)

    from collections import defaultdict
    metrics = defaultdict(float)
    for sample in tqdm(data, desc=f'Collecting data for {dataset_name}'):
        question = sample['question']
        # fact_before_filter = hipporag.query_to_fact(question, num_before_filter)
        fact_before_filter = hipporag.query_to_fact(question, num_before_filter)

        # if dataset_name.startswith('beir'):
        #     # shuffle facts_before_filter
        #     fact_before_filter = fact_before_filter[:1] + random.sample(fact_before_filter[1:], len(fact_before_filter) - 1)
        fact_before_filter = fact_before_filter[:num_before_filter]
        random.shuffle(fact_before_filter)

        gold_docs = get_gold_docs(dataset_name, sample)
        oracle_triples = get_oracle_triples(gold_docs, hipporag)
        assert len(gold_docs) > 0, f'No gold docs found for {dataset_name} query: {question}'
        if len(oracle_triples) == 0:
            print(f'No gold triples found for {dataset_name} query: {question}')

        # use triples from gold docs as facts after filtering
        fact_after_filter = [item for item in fact_before_filter if tuple(item) in oracle_triples]
        if fact_before_filter == fact_after_filter:
            metrics['num_same_before_after'] += 1
        metrics['num_fact_before_filter'] += len(fact_before_filter)
        metrics['num_fact_after_filter'] += len(fact_after_filter)

        res.append({'question': question, 'fact_before_filter': json.dumps({"fact": fact_before_filter}), 'fact_after_filter': json.dumps({"fact": fact_after_filter})})
    # end for each sample

    metrics['num_samples'] = len(data)
    metrics['num_fact_before_filter'] /= len(data)
    metrics['num_fact_after_filter'] /= len(data)
    return res, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--retriever', type=str)
    parser.add_argument('--linker', type=str)
    args = parser.parse_args()

    train_split = {'beir_msmarco_train_1000': 1000, 'musique_train_1000': 500, '2wikimultihopqa_train_1000': 500}
    dev_split = {'beir_msmarco_dev_1000': 1000, 'musique_dev_1000': 500, '2wikimultihopqa_dev_1000': 500}
    num_before_filter = 5

    os.makedirs('data/fact_filter', exist_ok=True)

    train_samples = []
    train_output_path = f'data/fact_filter/train.json'
    if not os.path.isfile(train_output_path):
        for dataset_name in train_split:
            num_sample = train_split[dataset_name]
            samples, metrics = collect_filter_data(dataset_name, num_sample, num_before_filter, args.extractor, args.retriever, args.linker)
            train_samples.extend(samples)
            print(f'{dataset_name}: {metrics}')
        with open(train_output_path, 'w') as f:
            json.dump(train_samples, f, indent=4)
            print(f'Saved {len(train_samples)} samples to {train_output_path}')
    else:
        print(f'File {train_output_path} already exists, skipping training data collection')

    dev_samples = []
    dev_output_path = f'data/fact_filter/dev.json'
    if not os.path.isfile(dev_output_path):
        for dataset_name in dev_split:
            num_sample = dev_split[dataset_name]
            samples, metrics = collect_filter_data(dataset_name, num_sample, num_before_filter, args.extractor, args.retriever, args.linker)
            dev_samples.extend(samples)
            print(f'{dataset_name}: {metrics}')
        with open(dev_output_path, 'w') as f:
            json.dump(dev_samples, f, indent=4)
            print(f'Saved {len(dev_samples)} samples to {dev_output_path}')
    else:
        print(f'File {dev_output_path} already exists, skipping dev data collection')
