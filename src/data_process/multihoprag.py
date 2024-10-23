import json

from datasets import load_dataset

if __name__ == '__main__':
    dataset_hf = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    corpus_hf = load_dataset("yixuantt/MultiHopRAG", "corpus")

    corpus = []
    title_set = set()
    url_set = set()
    for p in corpus_hf['train']:
        p['text'] = p['body']
        del p['body']
        corpus.append(p)
        title_set.add(p['title'])
        url_set.add(p['url'])

    dataset = []
    for sample in dataset_hf['train']:
        sample['question'] = sample['query']
        del sample['query']
        dataset.append(sample)

    with open('data/multihoprag.json', 'w') as f:
        json.dump(dataset, f)
    with open('data/multihoprag_corpus.json', 'w') as f:
        json.dump(corpus, f)

    print('dataset length:', len(dataset))
    print('corpus length:', len(corpus))
    print('title_set length:', len(title_set))
    print('url_set length:', len(url_set))
