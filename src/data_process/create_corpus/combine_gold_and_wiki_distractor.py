# 1. musique_full_wiki_subset
# 2. openie_corpus_batch

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100000)
    parser.add_argument('--llm', type=str, default='neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8', help='LLM model name')
    parser.add_argument('--distractor', type=str, default='output/corpus_batch', help='Path to distractor dir')
    args = parser.parse_args()

    llm_label = args.llm.replace('/', '_')
    dataset = json.load(open('data/musique.json'))
    corpus = json.load(open('data/musique_corpus.json'))
    existing_corpus_size = len(corpus)

    chunk_size = 10000
    done = False
    for batch_id in range(0, args.size // chunk_size + 1):
        batch_corpus = json.load(open(f'data/corpus_batch/musique_{batch_id}_corpus.json'))
        docs = batch_corpus['docs']
        for doc in docs:
            corpus.append({'idx': len(corpus), 'title': doc['title'], 'text': doc['text']})
            if len(corpus) > args.size:
                done = True
                break
        if done:
            break

    corpus_output_path = f'data/musique_wiki_{args.size}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
        print(f'Corpus saved to {corpus_output_path}, len: {len(corpus)}')

    dataset_output_path = f'data/musique_wiki_{args.size}.json'
    with open(dataset_output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        print(f'Dataset saved to {dataset_output_path}, len: {len(dataset)}')

    existing_openie_results = json.load(open(f'output/openie_musique_results_ner_{llm_label}_{existing_corpus_size}.json'))
    openie_output_path = 'output/openie_'

    query_ner_output_path = ''
