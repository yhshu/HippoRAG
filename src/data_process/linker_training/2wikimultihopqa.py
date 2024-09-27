import argparse
import json
import os
import random
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/2wikimultihopqa')
    args = parser.parse_args()

    os.makedirs('data/linker_training/queries', exist_ok=True)
    os.makedirs('data/linker_training/corpus', exist_ok=True)
    corpus_dict = {}
    full_text_to_id = {}
    corpus_id = 0
    num_sample = {'train': 20000, 'dev': 2000}
    random.seed(1)

    for split in num_sample.keys():
        path = os.path.join(args.dir, f'{split}.json')
        split_data = json.load(open(path, 'r'))

        if num_sample[split] is not None and isinstance(num_sample[split], int):
            assert 0 <= num_sample[split] <= len(split_data)
            split_data = random.sample(split_data, min(num_sample[split], len(split_data)))

        print(f'Processing {split} ({len(split_data)})')
        corpus = []
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            evidence_candidates = []

            if len(sample['supporting_facts']) != len(sample['evidences']):
                continue

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
                assert supporting_full_text is not None and sentence
                evidence_candidates.append(
                    {'passage_id': full_text_to_id.get(supporting_full_text, str(corpus_id)), 'sentence': sentence, 'triples': sample['evidences'][evidence_id],
                     'relevance': 'support'})

                if supporting_full_text in full_text_to_id:
                    continue

                # add to corpus
                corpus.append({'id': str(corpus_id), 'title': supporting_title, 'full_text': supporting_full_text})
                full_text_to_id[supporting_full_text] = str(corpus_id)
                corpus_id += 1

            sample['evidences'] = evidence_candidates
        # end for each sample

        corpus_dict.update({item['id']: item for item in corpus})

        output_path = f'data/linker_training/queries/2wikimultihopqa_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')
    # end for each split

    corpus_output_path = 'data/linker_training/corpus/2wikimultihopqa_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus_dict)}) to {corpus_output_path}')
