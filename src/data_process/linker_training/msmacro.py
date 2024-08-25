import argparse
import json
import os
import random

from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    args = parser.parse_args()

    os.makedirs('data/linker_training/queries', exist_ok=True)
    os.makedirs('data/linker_training/corpus', exist_ok=True)
    corpus_dict = {}
    full_text_to_id = {}
    corpus_id = 0
    num_sample = {'train': 5000, 'dev': 500}
    random.seed(1)

    dataset = load_dataset('ms_marco', 'v2.1')

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    for split in num_sample.keys():
        if split == 'train':
            split_data = train_dataset
        elif split == 'dev':
            split_data = validation_dataset
        elif split == 'test':
            split_data = test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")

        assert 0 <= num_sample[split] <= len(split_data)

        # random sample the split_data (HuggingFace dataset)
        split_data = split_data.shuffle(seed=1)
        split_data = split_data.select(range(num_sample[split]))

        corpus = []
        for sample in split_data:  # for each sample in the split_data
            evidence_candidates = []
            query = sample['query']
            query_id = sample['id']
            query_type = sample['query_type']

            for passsage_idx in range(0, len(sample['passages']['passage_text'])):
                is_selected = sample['passages']['is_selected'][passsage_idx]
                passage = sample['passages']['passage_text'][passsage_idx]
                url = sample['passages']['url'][passsage_idx]

                if is_selected:  # positive sample
                    evidence_candidates.append({'passage_id': full_text_to_id.get(passage, str(corpus_id)),
                                                'sentence': passage, 'triples': '', 'relevance': 'support',
                                                'query_type': query_type, 'url': url})

                    if passage in full_text_to_id:
                        continue

                    corpus.append({'id': str(corpus_id), 'title': query, 'text': passage, 'full_text': passage})
                    full_text_to_id[passage] = str(corpus_id)
                    corpus_id += 1

            sample['candidates'] = evidence_candidates
        # end of for each sample

        corpus_dict.update({item['id']: item for item in corpus})

        output_path = f'data/linker_training/queries/msmacro_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')
    # end for each split

    corpus_output_path = 'data/linker_training/corpus/msmacro_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus_dict)}) to {corpus_output_path}')
