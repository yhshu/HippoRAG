import sys
sys.path.append('.')
import json
import random

from src.data_process.util import generate_hash
from src.processing import query_data_has_duplication, corpus_has_duplication

def get_corpus_from_data(data):
    corpus = []
    corpus_content_hash_set = set()
    for sample in data:
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
    # end for each sample
    return corpus


def save_data(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)
        print(f'{len(data)} samples saved to {output_path}')

def save_corpus(corpus, output_path):
    with open(output_path, 'w') as f:
        json.dump(corpus, f)
        print(f'{len(corpus)} samples saved to {output_path}')


if __name__ == '__main__':
    data = json.load(open('data/nq-test-rear.json'))
    random.seed(1)
    random.shuffle(data)

    test_data = []
    dev_data = []
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
            if len(test_data) < 1000:
                test_data.append(sample)
            elif len(dev_data) < 1000:
                dev_data.append(sample)

    assert len(test_data) == 1000
    test_corpus = get_corpus_from_data(test_data)
    query_data_has_duplication(test_data, False, False)
    corpus_has_duplication(test_corpus)
    save_data(test_data, 'data/nq_rear.json')
    save_corpus(test_corpus, 'data/nq_rear_corpus.json')

    assert len(dev_data) == 1000
    dev_corpus = get_corpus_from_data(dev_data)
    query_data_has_duplication(dev_data, False, False)
    corpus_has_duplication(dev_corpus)
    save_data(dev_data, 'data/nq_rear_dev.json')
    save_corpus(dev_corpus, 'data/nq_rear_dev_corpus.json')