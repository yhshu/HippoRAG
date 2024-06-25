import argparse

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.RetrievalModule import RetrievalModule
from src.create_graph import create_graph
from src.named_entity_extraction_parallel import ner_parallel
from src.openie_with_retrieval_option_parallel import openie_with_retrieval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `sci_fact_test`, `fiqa_dev`.')
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--extractor', type=str, default='gpt-3.5-turbo', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--num_thread', type=int, default=10)
    parser.add_argument('--syn_thresh', type=float, default=0.8)

    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    openie_with_retrieval(args.dataset, args.run_ner, args.num_passages, args.llm, args.extractor, args.num_thread)
    ner_parallel(args.dataset, args.llm, args.extractor, args.num_thread)

    extraction_type = 'ner'
    processed_extractor_name = args.extractor.replace('/', '_')

    create_graph(args.dataset, extraction_type, processed_extractor_name, args.retriever, args.syn_thresh, False, True)
    retrieval_module = RetrievalModule(args.retriever, 'output/query_to_kb.tsv', 'mean')
    retrieval_module = RetrievalModule(args.retriever, 'output/kb_to_kb.tsv', 'mean')
    retrieval_module = RetrievalModule(args.retriever, 'output/rel_kb_to_kb.tsv', 'mean')
    create_graph(args.dataset, extraction_type, processed_extractor_name, args.retriever, args.syn_thresh, True, True)
