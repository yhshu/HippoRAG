import argparse
import copy
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def convert_hf_dataset_to_list(hf_dataset):
    dataset_list = []
    for i in range(len(hf_dataset)):
        dataset_list.append(hf_dataset[i])
    return dataset_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    os.makedirs('data/linker_training/queries', exist_ok=True)
    os.makedirs('data/linker_training/corpus', exist_ok=True)
    corpus_dict = {}
    full_text_to_id = {}
    corpus_id = 0
    num_sample = {'train': 20000, 'dev': 1000}
    random.seed(1)

    dataset = load_dataset('ms_marco', 'v2.1')  # Hugging Face dataset

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
        split_data = split_data.shuffle(seed=1)

        corpus = []
        split_data_list = []
        for sample_idx in tqdm(range(0, len(split_data))):  # for each sample in the split_data
            new_sample = copy.deepcopy(split_data[sample_idx])
            sample = split_data[sample_idx]
            evidence_candidates = []
            query = sample['query']
            query_id = str(sample['query_id'])
            query_type = sample['query_type']
            new_sample['id'] = query_id
            new_sample['question'] = query
            del new_sample['query']

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

            if len(evidence_candidates) > 0:  # if the query has at least one positive sample
                new_sample['candidates'] = evidence_candidates
                split_data_list.append(new_sample)
                if num_sample[split] is not None and isinstance(num_sample[split], int) and len(split_data_list) >= num_sample[split]:
                    break
        # end of for each sample

        corpus_dict.update({item['id']: item for item in corpus})

        output_path = f'data/linker_training/queries/msmacro_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data_list, f)
        print(f'Saving {split} ({len(split_data_list)}) to {output_path}')
    # end for each split

    corpus_output_path = 'data/linker_training/corpus/msmacro_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus_dict)}) to {corpus_output_path}')
