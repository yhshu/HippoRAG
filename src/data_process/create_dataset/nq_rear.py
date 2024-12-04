import json
import random

from src.data_process.util import generate_hash
from src.processing import query_data_has_duplication, corpus_has_duplication

if __name__ == '__main__':
    data = json.load(open('data/nq-test-rear.json'))
    random.seed(1)
    random.shuffle(data)

    sampled_data = []
    for sample in data:
        contexts = sample['ctxs']
        reference = sample['reference']

        has_supporting = False
        for c in contexts:
            # if any reference is in the context, then it is a supporting document
            for r in reference:
                if r.lower() in c.lower():
                    has_supporting = True
                    break
        if has_supporting:
            sampled_data.append(sample)

        if len(sampled_data) >= 1000:
            break

    corpus = []
    corpus_content_hash_set = set()
    for sample in sampled_data:
        contexts = sample['ctxs']
        reference = sample['reference']
        annotated_contexts = []
        for c in contexts:
            c_split = c.split(' Content: ')
            assert len(c_split) == 2
            title = c_split[0][7:]
            content = c_split[1]

            is_supporting= False
            for r in reference:
                if r.lower() in c.lower():
                    is_supporting = True
                    break
            annotated_contexts.append({'title': title, 'text': content, 'is_supporting': is_supporting})


            content_hash = generate_hash(title + '\n' + content)
            if content_hash not in corpus_content_hash_set:
                corpus_content_hash_set.add(content_hash)
                corpus.append({'idx': len(corpus), 'title': title, 'text': content})
        # end for each passage

        sample['contexts'] = annotated_contexts
        del sample['ctxs']

    query_data_has_duplication(sampled_data, False, False)
    corpus_has_duplication(corpus)

    data_output_path = 'data/nq_rear.json'
    with open(data_output_path, 'w') as f:
        json.dump(sampled_data, f)
        print(f'{len(sampled_data)} samples saved to {data_output_path}')

    corpus_output_path = 'data/nq_rear_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f)
        print(f'{len(corpus)} samples saved to {corpus_output_path}')
