import argparse
import json
import os.path
import random
from tqdm import tqdm

if __name__ == '__main__':
    """
    Collect queries samples by train/dev split and the corpus for the musique dataset.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    args = parser.parse_args()

    os.makedirs('data/linker_training/queries', exist_ok=True)
    os.makedirs('data/linker_training/corpus', exist_ok=True)
    corpus_dict = {}
    full_text_to_id = {}
    corpus_id = 0
    split_num_sample = {'train': 'all', 'dev': 'all'}  # specify the number of samples for each split
    random.seed(1)

    for split in split_num_sample.keys():
        path = os.path.join(args.dir, f'musique_ans_v1.0_{split}.jsonl')
        # read musique raw jsonl file
        split_data = []
        with open(path, 'r') as f:
            raw = f.readlines()
            for line in raw:
                split_data.append(json.loads(line))

        if split_num_sample[split] is not None and isinstance(split_num_sample[split], int):  # sample data
            assert 0 <= split_num_sample[split] <= len(split_data)
            split_data = random.sample(split_data, min(split_num_sample[split], len(split_data)))

        print(f'Processing {split} ({len(split_data)})')

        # add passages to corpus
        corpus = []
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            evidence_candidates = []
            for passage in sample['paragraphs']:
                # add to query data
                is_supporting = passage['is_supporting']
                if is_supporting is False:
                    continue
                full_text = passage['title'] + '\n' + passage['paragraph_text']

                evidence_candidates.append({'passage_id': full_text_to_id.get(full_text, str(corpus_id)),
                                            'sentence': passage['paragraph_text'], 'triples': '', 'relevance': 'support'})

                if full_text in full_text_to_id:
                    continue

                # add to corpus
                corpus.append({'id': str(corpus_id), 'title': passage['title'], 'text': passage['paragraph_text'], 'full_text': full_text})
                full_text_to_id[full_text] = str(corpus_id)
                corpus_id += 1

            sample['candidates'] = evidence_candidates

        corpus_dict.update({item['id']: item for item in corpus})

        output_path = f'data/linker_training/queries/musique_ans_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')
    # end for each split

    corpus_output_path = 'data/linker_training/corpus/musique_ans_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus_dict)}) to {corpus_output_path}')
