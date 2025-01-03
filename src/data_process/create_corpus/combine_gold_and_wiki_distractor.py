# 1. musique_full_wiki_subset
# 2. openie_corpus_batch

import argparse
import json
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100000)
    parser.add_argument('--llm', type=str, default='neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8',
                        help='LLM model name')
    parser.add_argument('--distractor', type=str, default='output/corpus_batch', help='Path to distractor dir')
    parser.add_argument('--dataset', type=str, default='musique')
    args = parser.parse_args()

    llm_label = args.llm.replace('/', '_')
    dataset = json.load(open(f'data/{args.dataset}.json'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))
    existing_corpus_size = len(corpus)

    corpus_content_set = set()
    for c in corpus:
        corpus_content_set.add(c['title'] + '\n' + c['text'])

    chunk_size = 10000
    done = False
    openie_docs = []
    for batch_id in range(0, args.size // chunk_size + 1):
        batch_corpus = json.load(open(f'output/corpus_batch/{args.dataset}_{batch_id}_corpus.json', 'r'))
        docs = batch_corpus['docs']
        for doc in docs:
            content = doc['title'].strip() + '\n' + doc['text'].strip()
            if content not in corpus_content_set:
                corpus_content_set.add(content)
                corpus.append({'title': doc['title'], 'text': doc['text']})
                openie_docs.append(doc)
            if len(corpus) >= args.size:
                done = True
                break
        if done:
            break
    new_size = len(corpus)

    dataset_label = f'{args.dataset}_wiki_{args.size}'
    existing_openie_results = json.load(open(f'output/openie_{args.dataset}_results_ner_{llm_label}_{existing_corpus_size}.json'))

    corpus_output_path = f'data/{dataset_label}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
        print(f'Corpus saved to {corpus_output_path}, len: {len(corpus)}')

    dataset_output_path = f'data/{dataset_label}.json'
    with open(dataset_output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        print(f'Dataset saved to {dataset_output_path}, len: {len(dataset)}')

    openie_output_path = f'output/openie_{dataset_label}_results_ner_{llm_label}_{new_size}.json'
    assert len(existing_openie_results['docs']) + len(openie_docs) == len(corpus), "Combined corpus length should match"
    openie_results = {
        'docs': existing_openie_results['docs'] + openie_docs,
        'ents_by_doc': existing_openie_results['ents_by_doc'] + [item.get('extracted_entities', []) for item in openie_docs],
        'avg_ent_chars': None,
        'avg_ent_words': None,
        'num_tokens': None,
        'approx_total_tokens': None,
    }
    with open(openie_output_path, 'w') as f:
        json.dump(openie_results, f, indent=2)
        print(f'OpenIE results saved to {openie_output_path}, len: {len(openie_results["docs"])}')

    query_ner_src_path = f'output/{args.dataset}_{llm_label}_queries.named_entity_output.tsv'
    query_ner_dest_path = f'output/{dataset_label}_{llm_label}_queries.named_entity_output.tsv'
    shutil.copy(query_ner_src_path, query_ner_dest_path)
    print(f'Query NER results saved to {query_ner_dest_path}')
