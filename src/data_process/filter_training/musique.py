import argparse
import json
import os
import random

from tqdm import tqdm

from src.data_process.util import generate_hash

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    split_num_sample = {'train': 10}
    random.seed(args.seed)

    for split in split_num_sample:
        full_text_hash_set = set()
        path = os.path.join(args.dir, f'musique_ans_v1.0_{split}.jsonl')

        split_data = []
        split_corpus = []
        with open(path, 'r') as f:
            raw = f.readlines()
            for line in raw:
                split_data.append(json.loads(line))

        split_size = split_num_sample[split] if split_num_sample[split] != 'all' else len(split_data)

        if split_num_sample[split] is not None and isinstance(split_num_sample[split], int):  # sample data
            assert 0 <= split_num_sample[split] <= len(split_data)
            split_data = random.sample(split_data, min(split_num_sample[split], len(split_data)))

        print(f'Processing {split} ({len(split_data)})')

        # add passages to corpus
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            for passage in sample['paragraphs']:
                # add to query data
                is_supporting = passage['is_supporting']

                full_text = passage['title'] + '\n' + passage['paragraph_text']
                full_text_hash = generate_hash(full_text)

                if full_text_hash in full_text_hash_set:
                    continue
                full_text_hash_set.add(full_text_hash)
                split_corpus.append({'idx': len(split_corpus), 'title': passage['title'], 'text': passage['paragraph_text']})
            # end for each passage
        # end for each sample

        corpus_output_path = f'data/musique_{split}_{split_size}_corpus.json'
        queries_output_path = f'data/musique_{split}_{split_size}.json'

        with open(corpus_output_path, 'w') as f:
            json.dump(split_corpus, f, indent=2)
            print(f'Corpus saved to {corpus_output_path}', len(split_corpus))
        with open(queries_output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
            print(f'Queries saved to {queries_output_path}', len(split_data))
    # end for each split