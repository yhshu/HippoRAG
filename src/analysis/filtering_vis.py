import argparse
import json

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str)
    args = parser.parse_args()

    log = json.load(open(args.log))
    num_same = 0
    num_triple_before = 0
    num_triple_after = 0
    num_no_triple_after = 0

    outputs = []
    for sample in tqdm(log):
        question = sample['question'] if 'question' in sample else sample['query']
        gold_doc = None
        if 'supporting_docs' in sample:
            gold_doc = sample['supporting_docs']
        if 'gold_passages' in sample:
            gold_doc = sample['gold_passages']
        fact_before_rerank = []
        fact_after_rerank = []
        if 'rerank' in sample:
            fact_before_rerank = sample['rerank']['facts_before_rerank']
            fact_after_rerank = sample['rerank']['facts_after_rerank']
            differences = set([tuple(item) for item in fact_before_rerank]) - set([tuple(item) for item in fact_after_rerank])

        if set([tuple(item) for item in fact_before_rerank]) == set([tuple(item) for item in fact_after_rerank]):
            num_same += 1
        num_triple_before += len(fact_before_rerank)
        num_triple_after += len(fact_after_rerank)
        num_no_triple_after += 1 if len(fact_after_rerank) == 0 else 0

        outputs.append(f"{question}\t{fact_before_rerank}\t{fact_after_rerank}\t{differences}\t{gold_doc}")

    print(f"Num samples: {len(log)}")
    print(f"Num same: {num_same / len(log)}")
    print(f"Num triple before filter: {num_triple_before / len(log)}")
    print(f"Num triple after filter: {num_triple_after / len(log)}")
    print(f"Num no triple after filter: {num_no_triple_after / len(log)}")

    output_path = args.log.replace('.json', '_rerank.tsv')
    with open(output_path, 'w') as f:
        f.write('\n'.join(outputs))
    print(f"Saved to {output_path}")
