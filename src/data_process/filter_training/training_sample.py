import argparse
import json
import os

from src.hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    hipporag = HippoRAG(args.dataset, 'openai', 'gpt-4o-mini', 'GritLM/GritLM-7B', 'ner', 'facts_and_sim_passage_node_unidirectional', 0.8, True, False, None, False, 'ppr', 0.5,
                        0.9, None, None, 'GritLM/GritLM-7B', None)

    res = []
    input_path = f'data/{args.dataset}.json'
    data = json.load(open(input_path))
    print(f'Loaded {len(data)} samples from {input_path}')
    num_same_before_after = 0
    for sample in data:
        question = sample['question']
        fact_before_filter = hipporag.query_to_fact(question, 5)

        gold_docs = []
        if args.dataset.startswith('2wikimultihopqa'):
            for item in sample['supporting_facts']:
                title = item[0]
                for c in sample['context']:
                    if c[0] == title:
                        gold_docs.append(c[0] + '\n' + ' '.join(c[1]))
                        break
        elif args.dataset.startswith('musique'):
            gold_docs = [item['title'] + '\n' + item['paragraph_text'] for item in sample['paragraphs'] if item['is_supporting']]
        elif args.dataset.startswith('beir'):
            gold_docs = [item['title'] + '\n' + item['text'] for item in sample['paragraphs']]

        oracle_triples = []
        for p in gold_docs:
            assert len(p) > 0 and '\n' in p
            oracle_triples += hipporag.get_triples_and_triple_ids_by_passage_content(p)[0]

        fact_after_filter = []
        if args.dataset.startswith('beir'):
            # use triples from gold docs as facts after filtering
            fact_after_filter = [item for item in fact_before_filter if tuple(item) in oracle_triples]
        if fact_before_filter == fact_after_filter:
            num_same_before_after += 1
        res.append({'question': question, 'fact_before_filter': fact_before_filter, 'fact_after_filter': fact_after_filter})
    # end for each sample

    os.makedirs('data/fact_filter', exist_ok=True)
    output_path = f'data/fact_filter/{args.dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)
        print(f'Saved {len(res)} samples to {output_path}')
    print(f'Num of same facts before and after filtering: {num_same_before_after}')
