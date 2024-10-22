import sys

sys.path.append('.')

import argparse

from src.RetrievalModule import RetrievalModule
from src.create_graph import create_graph
from src.named_entity_extraction_parallel import query_ner_parallel
from src.openie_with_retrieval_option_parallel import openie_for_corpus


def index_with_huggingface(dataset_name: str, run_ner: bool, num_passages, llm_provider: str, extractor: str, retriever: str,
                           num_thread, syn_thresh=0.8, langchain_db='.langchain.db', skip_openie=False, skip_graph=False,
                           passage_node=False):
    # set_llm_cache(SQLiteCache(database_path=langchain_db))
    if skip_openie is False:
        openie_for_corpus(dataset_name, run_ner, num_passages, llm_provider, extractor, num_thread)
    else:
        print('Skipping OpenIE')
    query_ner_parallel(dataset_name, llm_provider, extractor, num_thread)

    extraction_type = 'ner'
    processed_extractor_name = extractor.replace('/', '_')

    create_graph(dataset_name, extraction_type, processed_extractor_name, retriever, syn_thresh, False, True, passage_node)
    RetrievalModule(retriever, 'output/query_to_kb.tsv', 'mean')
    RetrievalModule(retriever, 'output/kb_to_kb.tsv', 'mean')
    RetrievalModule(retriever, 'output/rel_kb_to_kb.tsv', 'mean')
    create_graph(dataset_name, extraction_type, processed_extractor_name, retriever, syn_thresh, True, True, passage_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `scifact_test`, `fiqa_dev`.')
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--skip_openie', action='store_true')
    parser.add_argument('--skip_graph', action='store_true')
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--extractor', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--num_thread', type=int, default=10)
    parser.add_argument('--syn_thresh', type=float, default=0.8)
    parser.add_argument('--passage_node', type=str)

    args = parser.parse_args()
    assert args.passage_node is None or args.passage_node in ['unidirectional', 'bidirectional']
    index_with_huggingface(args.dataset, args.run_ner, args.num_passages, args.llm, args.extractor, args.retriever, args.num_thread, args.syn_thresh,
                           skip_openie=args.skip_openie, skip_graph=args.skip_graph, passage_node=args.passage_node)
