import sys
sys.path.append('.')

import argparse
import json
import os.path
import random

from tqdm import tqdm

from src.data_process.util import generate_hash

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/2wikimultihopqa')
    parser.add_argument('-ntrain', '--num_train', help='number of training samples', default=1000)
    parser.add_argument('-ndev', '--num_dev', help='number of dev samples', default=1000)
    parser.add_argument('-ntest', '--num_test', help='number of test samples')
    parser.add_argument('-src', '--source', help='source of the data, e.g., `train`, `dev`, `test`', default='train')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    split_num_sample = {}
    if args.num_train is not None:
        split_num_sample['train'] = args.num_train
    if args.num_dev is not None:
        split_num_sample['dev'] = args.num_dev
    if args.num_test is not None:
        split_num_sample['test'] = args.num_test

    path = os.path.join(args.dir, f'{args.source}.json')
    data = json.load(open(path, 'r'))
    random.seed(args.seed)
    random.shuffle(data)

    start_idx  = 0
    for split in split_num_sample:
        full_text_hash_set = set()
        split_corpus = []

        split_size = split_num_sample[split] if split_num_sample[split] != 'all' else len(data)

        if split_num_sample[split] is not None and isinstance(split_num_sample[split], int):
            assert start_idx + split_size <= len(data)
        split_data = data[start_idx:start_idx + split_size]
        start_idx += split_size

        print(f'Processing {split} ({len(split_data)}) from {args.source}[{start_idx - split_size}:{start_idx})')

        # add passages to corpus
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            for evidence_id in range(0, len(sample['supporting_facts'])):
                sentence = None
                supporting_full_text = None
                supporting_title = sample['supporting_facts'][evidence_id][0]
                sentence_id = sample['supporting_facts'][evidence_id][1]
                for context in sample['context']:
                    if context[0] == supporting_title:
                        supporting_full_text = ' '.join(context[1])
                        sentence = context[1][sentence_id]
                        break
                assert supporting_full_text is not None and sentence is not None

                full_text_hash = generate_hash(supporting_title + '\n' + supporting_full_text)
                if full_text_hash in full_text_hash_set:
                    continue
                full_text_hash_set.add(full_text_hash)
                split_corpus.append({'idx': len(split_corpus), 'title': supporting_title, 'text': supporting_full_text})

            for c in sample['context']:
                full_text = c[0] + '\n' + ' '.join(c[1])
                full_text_hash = generate_hash(full_text)
                if full_text_hash in full_text_hash_set:
                    continue
                full_text_hash_set.add(full_text_hash)
                split_corpus.append({'idx': len(split_corpus), 'title': c[0], 'text': ' '.join(c[1])})
            # end for each evidence
        # end for each sample

        corpus_output_path = f'data/2wikimultihopqa_{split}_{len(split_data)}_corpus.json'
        queries_output_path = f'data/2wikimultihopqa_{split}_{len(split_data)}.json'

        with open(corpus_output_path, 'w') as f:
            json.dump(split_corpus, f, indent=2)
            print(f'Corpus saved to {corpus_output_path}', len(split_corpus))
        with open(queries_output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
            print(f'Queries saved to {queries_output_path}', len(split_data))

    # end for each split
