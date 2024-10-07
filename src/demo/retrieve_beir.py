# Note that BEIR uses https://github.com/cvangysel/pytrec_eval to evaluate the retrieval results.
import sys

sys.path.append('.')

from collections import defaultdict
from typing import Union

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.data_process.util import merge_chunk_scores, merge_chunks
from src.hipporag import HippoRAG
import os
import pytrec_eval
import argparse
import json
from tqdm import tqdm


def detailed_log(dataset: list, run_dict, eval_res, chunk=False, threshold=None, dpr_only=False):
    logs = []
    for idx, query_id in tqdm(enumerate(run_dict['retrieved']), desc='Error analysis', total=len(run_dict['retrieved'])):
        item = dataset[idx]
        if threshold is not None and eval_res[query_id]['ndcg'] >= threshold:
            continue
        gold_passages = dataset[idx]['paragraphs']
        gold_passage_ids = [p['idx'] for p in item['paragraphs']]

        distances = []
        num_dis = 0
        gold_passage_extracted_entities = []
        gold_passage_extracted_triples = []
        if not dpr_only:
            gold_passage_extractions = [hipporag.get_raw_extraction_by_passage_idx(p_idx, chunk) for p_idx in gold_passage_ids]
            gold_passage_extracted_entities = [e for extr in gold_passage_extractions for e in extr['extracted_entities']]
            gold_passage_extracted_triples = [t for extr in gold_passage_extractions for t in extr['extracted_triples']]

            if 'linked_node_scores' in run_dict['log'][query_id]:
                linked_node_scores = run_dict['log'][query_id].get('linked_node_scores', '{}')
                if isinstance(linked_node_scores, str):
                    linked_node_scores = json.loads(linked_node_scores)
                for node_linking in linked_node_scores:
                    if isinstance(node_linking, list):
                        linked_node_phrase = node_linking[1]
                    else:  # node_linking is a string as a key of linked_node_scores
                        linked_node_phrase = linked_node_scores[node_linking]
                    distance = []
                    for e in gold_passage_extracted_entities:
                        if e == linked_node_phrase:
                            distance.append(0)
                            num_dis += 1
                        d = hipporag.get_shortest_distance_between_nodes(linked_node_phrase, e)
                        if d > 0:
                            distance.append(d)
                            num_dis += 1
                    distances.append(distance)

        pred_passages = []
        for pred_corpus_id in run_dict['retrieved'][query_id]:
            if not chunk:
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id:
                        pred_passages.append(corpus_item)
            else:
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id or corpus_item['idx'].startswith(f'{pred_corpus_id}_'):
                        pred_passages.append(corpus_item)
        if chunk:
            pred_passages = merge_chunks(pred_passages)

        log = {
            'query': dataset[idx]['text'],
            'ndcg@10': eval_res[query_id]['ndcg_cut_10'],
            'gold_passages': gold_passages,
            'pred_passages': pred_passages,
            'log': run_dict['log'][query_id],
            'distances': distances,
            'avg_distance': sum([sum(d) for d in distances]) / num_dis if num_dis > 0 else None,
            'entities_in_supporting_passage': gold_passage_extracted_entities,
            'triples_in_supporting_passage': gold_passage_extracted_triples,
        }
        for key in run_dict['log'][query_id]:
            if key not in log:
                log[key] = run_dict['log'][query_id][key]
        logs.append(log)
    # end for each query
    return logs


def run_retrieve_beir(dataset_name: str, extractor_name: str, retriever_name: str, linker_name: str, linking: str,
                      doc_ensemble: bool, dpr_only: bool, chunk: bool, link_top_k: Union[int, None] = 3, oracle_extraction=False, reranker_name=None):
    doc_ensemble_str = 'doc_ensemble' if doc_ensemble else 'no_ensemble'
    extraction_str = extractor_name.replace('/', '_').replace('.', '_')
    graph_creating_str = retriever_name.replace('/', '_').replace('.', '_')
    if linker_name is None:
        linker_name = retriever_name
    linking_str = f"{linker_name.replace('/', '_').replace('.', '_')}_linking_{linking}_top_k_{link_top_k}"
    if oracle_extraction:
        linking_str += '_oracle_ie'
    dpr_only_str = '_dpr_only' if dpr_only else ''
    reranker_str = f'_RE_{reranker_name}' if reranker_name is not None else ''
    os.makedirs(f'output/retrieval/{dataset_name}', exist_ok=True)
    run_output_path = f'output/retrieval/{dataset_name}/{dataset_name}_run_{doc_ensemble_str}_E_{extraction_str}_R_{graph_creating_str}_L_{linking_str}{dpr_only_str}{reranker_str}.json'
    print(f'Log will be saved to {run_output_path}')  # this file is used for pytrec_eval, another log file will be saved for details

    pytrec_metrics = {'map_cut_10', 'ndcg_cut_10'}
    metrics = defaultdict(float)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_metrics)
    if os.path.isfile(run_output_path):
        run_dict = json.load(open(run_output_path))
        print(f'Log file found at {run_output_path}, len: {len(run_dict["retrieved"])}')
    else:
        run_dict = {'retrieved': {}, 'log': {}}  # for pytrec_eval

    to_update_run = False
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc='Evaluating samples'):
        query_text = sample['text']
        query_id = sample['id']
        # if query_id in run_dict['retrieved']:
        #     continue
        supporting_docs = sample['paragraphs']
        if oracle_extraction or hipporag.reranker_name in ['oracle_triple']:
            oracle_triples = []
            for p in supporting_docs:
                oracle_triples += hipporag.get_triples_and_triple_ids_by_corpus_idx(hipporag.get_corpus_idx_by_passage_idx(p['idx']))[0]
        else:
            oracle_triples = None
        ranks, scores, log = hipporag.rank_docs(query_text, doc_top_k=10, link_top_k=link_top_k, linking=linking, oracle_triples=oracle_triples)

        retrieved_docs = [corpus[r] for r in ranks]
        to_update_run = True

        if linking in ['ner_to_node', 'query_to_node', 'query_to_fact', 'query_to_passage']:  # evaluate the recall of nodes from supporting documents
            # get oracle nodes
            if oracle_triples is None:
                oracle_triples = []
                for p in supporting_docs:
                    oracle_triples += hipporag.get_triples_and_triple_ids_by_corpus_idx(hipporag.get_corpus_idx_by_passage_idx(p['idx']))[0]
            oracle_nodes = set([t[0] for t in oracle_triples]).union(set([t[2] for t in oracle_triples]))

            # get linked nodes
            linked_nodes = set()
            if log is not None and len(log) and 'linked_node_scores' in log:
                log['nodes_in_supporting_doc'] = list(oracle_nodes)
                for item in log['linked_node_scores']:
                    assert isinstance(item, list) or isinstance(item, str)
                    if isinstance(item, list):  # item: mention -> node phrase
                        linked_nodes.add(item[1])
                    elif isinstance(item, str):
                        linked_nodes.add(item)
                    if link_top_k is not None and len(linked_nodes) >= link_top_k:
                        break
                linked_nodes = set([node for node in linked_nodes if '\n' not in node])  # remove passage nodes

                # calculate recall
                node_precision = len(oracle_nodes.intersection(set(linked_nodes))) / len(linked_nodes) if len(linked_nodes) > 0 else 0
                node_recall = len(oracle_nodes.intersection(set(linked_nodes))) / len(oracle_nodes) if len(oracle_nodes) > 0 else 0
                node_hit = True if len(oracle_nodes.intersection(set(linked_nodes))) > 0 else False
                metrics['node_precision'] += node_precision
                metrics['node_recall'] += node_recall
                metrics['node_hit'] += node_hit
            else:
                hipporag.logger.info(f'No linked nodes found for query {query_id}.')

        # evaluate the retrieval results
        log['query'] = query_text
        run_dict['retrieved'][query_id] = {doc['idx']: score for doc, score in zip(retrieved_docs, scores)}
        run_dict['log'][query_id] = log
    # end each query

    if to_update_run:  # if there are new results from this run, save them
        with open(run_output_path, 'w') as f:
            json.dump(run_dict, f)
            print(f'Run saved to {run_output_path}, len: {len(run_dict["retrieved"])}')

    # postprocess run_dict['retrieved'] if the corpus is chunked
    if chunk:
        for idx in run_dict['retrieved']:
            run_dict['retrieved'][idx] = merge_chunk_scores(run_dict['retrieved'][idx])
    eval_res = evaluator.evaluate(run_dict['retrieved'])

    # get average scores
    avg_scores = {}
    for metric in pytrec_metrics:
        avg_scores[metric] = round(sum([v[metric] for v in eval_res.values()]) / len(eval_res), 3)
    print(f'Evaluation results: {avg_scores}')

    for key in metrics:
        metrics[key] /= len(dataset) if len(dataset) > 0 else 1
        metrics[key] = round(metrics[key], 3)
    print(f'Metrics: {metrics}')
    print(hipporag.statistics)

    logs = detailed_log(dataset, run_dict, eval_res, chunk, dpr_only=dpr_only)
    os.makedirs(f'output/retrieval/{dataset_name}', exist_ok=True)
    detailed_log_output_path = f'output/retrieval/{dataset_name}/{dataset_name}_log_{doc_ensemble_str}_E_{extraction_str}_R_{graph_creating_str}_L_{linking_str}{dpr_only_str}{reranker_str}.json'
    with open(detailed_log_output_path, 'w') as f:
        json.dump(logs, f)
    print(f'Detailed log saved to {detailed_log_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `scifact_test`, `fiqa_dev`.')
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--extractor', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--retriever', type=str, help="Graph creating retriever name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--linker', type=str, help="Node linking model name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--link_top_k', type=int, default=5)
    parser.add_argument('--reranker', type=str)
    parser.add_argument('--linking', type=str)
    parser.add_argument('--doc_ensemble', action='store_true')
    parser.add_argument('--dpr_only', action='store_true')
    parser.add_argument('--oracle_ie', action='store_true')
    parser.add_argument('--num', help='the number of samples to evaluate')
    parser.add_argument('-rs', '--recognition_threshold', type=float, default=0.9)
    parser.add_argument('--graph_type', type=str, default='facts_and_sim')
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--directed', action='store_true')
    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=f".hipporag_{args.extractor}.db"))

    if args.chunk is False and 'chunk' in args.dataset:
        args.chunk = True
    # assert at most only one of them is True
    assert not (args.doc_ensemble and args.dpr_only)
    try:
        corpus_path = f'data/{args.dataset}_corpus.json'
        print('Loading corpus from ', corpus_path)
        corpus = json.load(open(corpus_path, 'r'))

        qrel_path = f'data/{args.dataset}_qrel.json'
        print('Loading qrel from ', qrel_path)
        qrel = json.load(open(qrel_path, 'r'))  # note that this is json file processed from tsv file, used for pytrec_eval
    except Exception as e:
        print(f'Error when loading files: {e}')
        exit(1)
    with open(f'data/{args.dataset}.json') as f:
        dataset = json.load(f)

    if args.num:
        dataset = dataset[:min(int(args.num), len(dataset))]
        qrel = {key: qrel[key] for i, key in enumerate(qrel) if i < min(int(args.num), len(dataset))}

    hipporag = HippoRAG(args.dataset, 'openai', args.extractor, args.retriever, doc_ensemble=args.doc_ensemble,
                        dpr_only=args.dpr_only, linker_name=args.linker, recognition_threshold=args.recognition_threshold, reranker_name=args.reranker,
                        graph_type=args.graph_type, damping=args.damping, directed_graph=args.directed)

    if args.linking == 'ner_to_node':
        link_top_k = None
    run_retrieve_beir(args.dataset, args.extractor, args.retriever, args.linker, args.linking,
                      args.doc_ensemble, args.dpr_only, args.chunk, link_top_k=args.link_top_k, oracle_extraction=args.oracle_ie, reranker_name=args.reranker)
