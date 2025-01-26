import sys

sys.path.append('.')

import argparse
from collections import defaultdict
import json
import os

from tqdm import tqdm
from transformers.hf_argparser import string_to_bool

from src.hipporag import HippoRAG
from src.ircot_hipporag import get_gold_docs, get_oracle_triples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--linker', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--reranker', type=str)
    parser.add_argument('--linking', type=str, default='query_to_fact', help='linking method for the entry point of the graph')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of demo samples')
    parser.add_argument('--max_steps', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    parser.add_argument('--link_top_k', type=int, help='the number of linked nodes at each retrieval step')
    parser.add_argument('--doc_ensemble', type=str, default='f')
    parser.add_argument('--dpr_only', action='store_true')
    parser.add_argument('--graph_alg', type=str, default='ppr')
    parser.add_argument('--graph_type', type=str, default='facts_and_sim_passage_node_unidirectional')
    parser.add_argument('--wo_node_spec', action='store_true')
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    parser.add_argument('--recognition_threshold', type=float, default=0.9)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--force_retry', action='store_true')
    parser.add_argument('--directed', action='store_true')
    args = parser.parse_args()

    # process args
    doc_ensemble = string_to_bool(args.doc_ensemble)
    doc_ensemble_str = f'doc_ensemble_{args.recognition_threshold}' if doc_ensemble else 'no_ensemble'
    dpr_only_str = 'dpr_only' if args.dpr_only else 'hipporag'
    llm_model_name_processed = args.llm_model.replace('/', '_').replace('.', '_')
    rerank_model_name_processed = args.reranker.replace('/', '_').replace('.', '_') if args.reranker else ''
    rerank_str = f'_RE_{rerank_model_name_processed}' if rerank_model_name_processed != '' else ''
    graph_type_str = ''
    if 'passage_node' in args.graph_type:
        graph_type_str = '_GT_pn'
        if 'unidirectional' in args.graph_type:
            graph_type_str += 'u'

    hipporag = HippoRAG(args.dataset, extraction_model=args.llm, extractor_name=args.llm_model, graph_creating_retriever_name=args.retriever,
                        linker_name=args.linker,
                        doc_ensemble=doc_ensemble, node_specificity=not (args.wo_node_spec), sim_threshold=args.sim_threshold,
                        dpr_only=args.dpr_only, graph_alg=args.graph_alg, damping=args.damping, recognition_threshold=args.recognition_threshold,
                        reranker_name=args.reranker, graph_type=args.graph_type, directed_graph=args.directed)
    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    max_steps = args.max_steps

    output_path = (
        f'output/query_to_fact/{args.dataset}/{args.dataset}_{dpr_only_str}{graph_type_str}_E_{llm_model_name_processed}_R_{hipporag.graph_creating_retriever_name_processed}_L_{hipporag.linking_retriever_name_processed}_{args.linking}{rerank_str}'
        f'_demo_{args.num_demo}_step_{max_steps}_top_{args.top_k}_{args.graph_alg}_damp_{args.damping}_sim_{args.sim_threshold}')

    os.makedirs(f'output/query_to_fact/{args.dataset}', exist_ok=True)

    if args.wo_node_spec:
        output_path += 'wo_node_spec'
    if args.link_top_k:
        output_path += f'_LT_{args.link_top_k}'
    output_path += '.json'
    print('Log file will be saved to', output_path)
    hipporag.load_triple_vectors()

    k_list = [1, 5, 10, 20, 30, 50, 80, 100, 150, 200]
    metrics = defaultdict(float)

    fact_comparison = []
    for sample_idx, sample in tqdm(enumerate(data), total=len(data), desc='Retrieval'):  # for each sample
        if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample_id = sample['_id']
        elif 'id' in sample:
            sample_id = sample['id']
        else:
            sample_id = str(sample_idx)

        query = sample['question']

        linked_facts = hipporag.query_to_fact(query, max(k_list))
        facts_in_gold_passage = []

        gold_docs = get_gold_docs(args.dataset, sample)
        oracle_facts = get_oracle_triples(gold_docs, hipporag)

        # calculate recall@k using linked_facts and oracle_facts, recall@k: the number of correct facts in top k linked facts
        for k in k_list:
            recall_at_k = 0
            for i, fact in enumerate(linked_facts[:k]):
                if fact in oracle_facts:
                    recall_at_k += 1
            if len(oracle_facts) == 0:
                recall_at_k = 0
            else:
                recall_at_k /= len(oracle_facts)
            metrics[f'R@{k}'] += recall_at_k
            print(f"R@{k}: {round(metrics[f'R@{k}'] / (sample_idx + 1), 4)}", end=' ')

            assert k != 0
            precision_at_k = 0
            for i, fact in enumerate(linked_facts[:k]):
                if fact in oracle_facts:
                    precision_at_k += 1
            precision_at_k /= k
            metrics[f'P@{k}'] += precision_at_k
            print(f"P@{k}: {round(metrics[f'P@{k}'] / (sample_idx + 1), 4)}", end=' ')
            print()
        print()

        fact_comparison.append({
            'query': query,
            'linked_facts': linked_facts[:20],
            'oracle_facts': oracle_facts
        })

    for k in k_list:
        metrics[f'R@{k}'] /= len(data)
        print(f'R@{k}: {round(metrics[f"R@{k}"], 4)}')
        metrics[f'P@{k}'] /= len(data)
        print(f'P@{k}: {round(metrics[f"P@{k}"], 4)}')

    with open(f'output/query_to_fact/{args.dataset}/query_to_fact.json', 'w') as f:
        json.dump(fact_comparison, f, indent=2)