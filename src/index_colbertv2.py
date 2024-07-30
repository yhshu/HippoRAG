import sys

sys.path.append('.')

import argparse

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.colbertv2_indexing import colbertv2_graph_indexing
from src.colbertv2_knn import colbertv2_retrieve_knn
from src.create_graph import create_graph
from src.named_entity_extraction_parallel import query_ner_parallel
from src.openie_with_retrieval_option_parallel import openie_for_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `scifact_test`, `fiqa_dev`.')
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--extractor', type=str, default='gpt-3.5-turbo', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='colbertv2')
    parser.add_argument('--num_thread', type=int, default=10)
    parser.add_argument('--syn_thresh', type=float, default=0.8)
    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    extraction_type = 'ner'

    # Running Open Information Extraction
    openie_for_corpus(args.dataset, args.run_ner, args.num_passages, args.llm, args.extractor, args.num_thread)
    query_ner_parallel(args.dataset, args.llm, args.extractor, args.num_thread)

    # Creating ColBERT Graph
    create_graph(args.dataset, extraction_type, args.extractor, args.retriever, args.syn_thresh, False, True)

    # Getting Nearest Neighbor Files
    colbertv2_retrieve_knn('output/kb_to_kb.tsv')
    colbertv2_retrieve_knn('output/query_to_kb.tsv')

    create_graph(args.dataset, extraction_type, args.extractor, args.retriever, args.syn_thresh, True, True)

    # ColBERTv2 Indexing for Entity Retrieval & Ensembling
    colbertv2_graph_indexing(args.dataset, f'data/{args.dataset}_corpus.json', f'output/{args.dataset}_facts_and_sim_graph_phrase_dict_ents_only_lower_preprocess_ner.v3.subset.p')
