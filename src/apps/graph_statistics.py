import sys

sys.path.append('.')

import argparse
import json

from src.hipporag import HippoRAG
from src.util import string_to_bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='vllm')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--linker', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--reranker', type=str)
    parser.add_argument('--linking', type=str, default='query_to_fact', help='linking method for the entry point of the graph')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of demo samples')
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

    num_passage = hipporag.docs_to_phrases_mat.shape[0]
    num_phrase_node = hipporag.docs_to_phrases_mat.shape[1]


    def count_edges_with_passage_nodes(g, threshold):
        count = sum(1 for e in g.es if e.source >= threshold or e.target >= threshold)
        return count

    num_edge_with_passage_node = count_edges_with_passage_nodes(hipporag.g, num_phrase_node)
    num_node = hipporag.g.vcount()
    num_edge = hipporag.g.ecount()
    print('=' * 50)
    print(args.dataset, args.llm_model)
    print(f'#phrase_nodes: {num_phrase_node}')
    print(f"#passage_nodes: {num_passage}")
    print(f'#total_nodes: {len(hipporag.kb_node_phrase_to_id)}')
    print(f'#unique_extracted_triple: {len(set(hipporag.triples))}')
    print(f'#synonym_edge: {num_edge - num_edge_with_passage_node - len(hipporag.triples)}')
    print(f'#edges with passage nodes: {num_edge_with_passage_node}')
    print(f'#total_edges: {num_edge}')
    print('=' * 50)
