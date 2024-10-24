import json

from datasets import load_dataset

if __name__ == '__main__':
    dataset_hf = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    corpus_hf = load_dataset("yixuantt/MultiHopRAG", "corpus")

    corpus = []
    title_to_text = dict()
    url_set = set()
    no_evidence = 0
    for p in corpus_hf['train']:
        p['text'] = p['body']
        del p['body']
        corpus.append(p)
        title_to_text[p['title']] = p['text']
        url_set.add(p['url'])

    dataset = []
    for sample in dataset_hf['train']:
        sample['question'] = sample['query']
        sample['paragraphs'] = sample['evidence_list']
        del sample['query']
        del sample['evidence_list']

        for p in sample['paragraphs']:
            p['text'] = title_to_text[p['title']]
            p['is_supporting'] = True
        if len(sample['paragraphs']) == 0:
            no_evidence += 1
            continue
        dataset.append(sample)

    with open('data/multihoprag.json', 'w') as f:
        json.dump(dataset, f)
    with open('data/multihoprag_corpus.json', 'w') as f:
        json.dump(corpus, f)

    print('dataset_hf length:', len(dataset_hf['train']))
    print('dataset length:', len(dataset))
    print('corpus length:', len(corpus))
    print('title_set length:', len(title_to_text))
    print('url_set length:', len(url_set))
    print('no_evidence:', no_evidence)