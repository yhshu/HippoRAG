import argparse
import json
import os.path
import random

from tqdm import tqdm

from src.data_process.util import generate_hash

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/2wikimultihopqa')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    split_num_sample = {'train': 10, 'dev': None}
    random.seed(args.seed)

    for split in split_num_sample:
        full_text_hash_set = set()
        path = os.path.join(args.dir, f'{split}.json')
        split_data = json.load(open(path, 'r'))

        split_data = []
        split_corpus = []

        path = os.path.join(args.dir, f'{split}.json')
        split_data = json.load(open(path, 'r'))

        if split_num_sample[split] is not None and isinstance(split_num_sample[split], int):
            assert 0 < split_num_sample[split] <= len(split_data)
            split_data = random.sample(split_data, min(split_num_sample[split], len(split_data)))

        print(f'Processing {split} ({len(split_data)})')

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
            # end for each evidence
        # end for each sample

        corpus_output_path = f'data/2wikimultihopqa_{split}_{split_num_sample[split]}_corpus.json'
        queries_output_path = f'data/2wikimultihopqa_{split}_{split_num_sample[split]}.json'

        with open(corpus_output_path, 'w') as f:
            json.dump(split_corpus, f, indent=2)
            print(f'Corpus saved to {corpus_output_path}', len(split_corpus))
        with open(queries_output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
            print(f'Queries saved to {queries_output_path}', len(split_data))

    # end for each split
