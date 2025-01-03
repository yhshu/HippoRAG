import argparse
import json
import os
import random

from tqdm import tqdm

from src.data_process.util import generate_hash
from src.processing import query_data_has_duplication, corpus_has_duplication

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/hotpotqa')
    parser.add_argument('-ntrain', '--num_train', help='number of training samples', default=1000)
    parser.add_argument('-ndev', '--num_dev', help='number of dev samples', default=1000)
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

    if args.source == 'train':
        data = json.load(open(os.path.join(args.dir, 'hotpot_train_v1.1.json')))
    elif args.source == 'dev':
        data = json.load(open(os.path.join(args.dir, 'hotpot_dev_distractor_v1.json')))
    else:
        assert False, f'Invalid data source: {args.source}'

    # todo: unfinished
    random.seed(args.seed)
    random.shuffle(data)

    for split in split_num_sample:
        full_text_hash_set = set()
        split_corpus = []
        if split == 'train':
            split_data = train_data
        elif split == 'dev':
            split_data = dev_data
        else:
            raise ValueError(f'Invalid split: {split}')

        if split_num_sample[split] is not None and isinstance(split_num_sample[split], int):
            assert 0 < split_num_sample[split] <= len(split_data)
            split_data = random.sample(split_data, min(split_num_sample[split], len(split_data)))
        else:
            continue

        print(f'Processing {split} ({len(split_data)})')

        # add passages to corpus
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            for c in sample['context']:
                full_text = c[0] + '\n' + ''.join(c[1])
                full_text_hash = generate_hash(full_text)
                if full_text_hash in full_text_hash_set:
                    continue
                full_text_hash_set.add(full_text_hash)
                split_corpus.append({'idx': len(split_corpus), 'title': c[0], 'text': ''.join(c[1])})

        # end for each sample

        data_output_path = f'data/hotpotqa_{split}_{len(split_data)}.json'
        corpus_output_path = f'data/hotpotqa_{split}_{len(split_data)}_corpus.json'

        duplication, d = query_data_has_duplication(split_data, True, False)
        if duplication:
            print(f"{split} has query duplication, #query: {len(d)}")
        duplication = corpus_has_duplication(split_corpus)
        if duplication:
            print(f"{split} corpus has passage duplication")

        with open(data_output_path, 'w') as f:
            json.dump(split_data, f, indent=4)
            print(f'Saved {len(split_data)} samples to {data_output_path}')

        with open(corpus_output_path, 'w') as f:
            json.dump(split_corpus, f, indent=4)
            print(f'Saved {len(split_corpus)} samples to {corpus_output_path}')
    # end for each split
