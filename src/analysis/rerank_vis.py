import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str)
    args = parser.parse_args()

    log = json.load(open(args.log))
    for sample in log:
        print(sample['question'] if 'question' in sample else sample['query'])
        if 'supporting_docs' in sample:
            print(sample['supporting_docs'])
        if 'gold_passages' in sample:
            print(sample['gold_passages'])
        print(sample['rerank']['facts_before_rerank'])
        print(sample['rerank']['facts_after_rerank'])
        print()