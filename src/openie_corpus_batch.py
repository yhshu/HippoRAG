import json
import os
import sys

from tqdm import tqdm

from src.langchain_util import init_llm_client
from src.processing import fix_broken_generated_json

sys.path.append('.')

import argparse
from src.openie_with_retrieval_option_parallel import openie_for_corpus, named_entity_recognition_batch_vllm, openie_post_ner_extract_batch_vllm


def openie(dataset_name: str, run_ner: bool, num_passages, llm_provider: str, extractor: str, retriever: str,
           num_thread, num_gpus=4):
    from src.langchain_util import init_llm_client
    gpu_mem_util = 0.95
    if llm_provider == 'vllm':
        client = init_llm_client(llm_provider, extractor, num_gpus=num_gpus, gpu_memory_utilization=gpu_mem_util)  # LangChain model
    else:
        client = init_llm_client(llm_provider, extractor)
    openie_for_corpus(dataset_name, run_ner, num_passages, llm_provider, extractor, num_thread, client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='data/corpus_batch/musique_0_corpus.json', help='corpus path')
    parser.add_argument('--llm', type=str, default='vllm', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--extractor', type=str, default='neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8', help='Specific model name')
    parser.add_argument('--num_gpus', type=int, default=4)

    args = parser.parse_args()
    print(args)

    corpus_label = args.corpus.split('/')[-1].split('.')[0]
    print(f'Processing corpus: {corpus_label}')
    client = init_llm_client(args.llm, args.extractor, num_gpus=args.num_gpus)
    corpus = json.load(open(args.corpus, 'r'))

    passages = []
    for p in corpus:
        content = p['title'] + '\n' + p['text']
        passages.append(content)
    ner_responses, ner_total_tokens = named_entity_recognition_batch_vllm(client, passages, args.extractor)

    entities_list = []
    for r in ner_responses:
        if len(r) == 0:
            print('Empty NER response')
            entities_list.append([])
            continue
        try:
            entities_list.append(r.get('named_entities', []))
        except Exception as e:
            print('Error:', e)
            entities_list.append([])
    assert len(entities_list) == len(passages)
    openie_responses, openie_total_tokens = openie_post_ner_extract_batch_vllm(client, passages, entities_list, args.extractor)
    triples = []
    for r in openie_responses:
        if len(r) == 0:
            print('Empty OpenIE response')
            triples.append([])
            continue
        try:
            r = json.loads(r)
            triples.append(r.get('triples', []))
        except Exception as e:
            try:
                r = fix_broken_generated_json(r)
                r = json.loads(r)
                triples.append(r.get('triples', []))
            except Exception as e:
                triples.append([])

    assert len(triples) == len(passages)

    docs = []
    ents_by_doc = []
    avg_ent_chars = 0
    avg_ent_words = 0
    num_entity = 0
    num_completion_tokens = sum(ner_total_tokens) + sum(openie_total_tokens)
    for p_idx, p in tqdm(enumerate(corpus), desc="Finishing"):
        docs.append({"title": p['title'], "text": p['text'], "passage": p['title']+ '\n' + p['text'],
                     "extracted_entities": entities_list[p_idx], "extracted_triples": triples[p_idx]})
        ents_by_doc.append(entities_list[p_idx])
        avg_ent_chars += sum([len(e) for e in entities_list[p_idx]])
        avg_ent_words += sum([len(e.split()) for e in entities_list[p_idx]])
        num_entity += len(entities_list[p_idx])

    avg_ent_chars /= num_entity if num_entity > 0 else 1
    avg_ent_words /= num_entity if num_entity > 0 else 1

    openie_results = {
        "docs": docs,
        "ents_by_doc": ents_by_doc,
        "avg_ent_chars": avg_ent_chars,
        "avg_ent_words": avg_ent_words,
        "num_tokens": num_completion_tokens,
        "approx_total_tokens": num_completion_tokens,
    }

    os.makedirs('output/corpus_batch', exist_ok=True)
    with open(f'output/corpus_batch/{corpus_label}.json', 'w') as f:
        json.dump(openie_results, f)
