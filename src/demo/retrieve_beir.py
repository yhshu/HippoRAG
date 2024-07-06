# Note that BEIR uses https://github.com/cvangysel/pytrec_eval to evaluate the retrieval results.
import sys
from collections import defaultdict

sys.path.append('.')

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
                for node_linking in run_dict['log'][query_id]['linked_node_scores']:
                    linked_node_phrase = node_linking[1]
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

        logs.append({
            'query': dataset[idx]['text'],
            'ndcg': eval_res[query_id]['ndcg'],
            'gold_passages': gold_passages,
            'pred_passages': pred_passages,
            'log': run_dict['log'][query_id],
            'distances': distances,
            'avg_distance': sum([sum(d) for d in distances]) / num_dis if num_dis > 0 else None,
            'entities_in_supporting_passage': gold_passage_extracted_entities,
            'triples_in_supporting_passage': gold_passage_extracted_triples,
        })
    return logs


def run_retrieve_beir(dataset_name: str, extraction_model: str, retrieval_model: str, linking_model: str, linking: str, doc_ensemble: bool, dpr_only: bool, chunk: bool,
                      detail: bool,
                      link_top_k=3, oracle_extraction=False):
    doc_ensemble_str = 'doc_ensemble' if doc_ensemble else 'no_ensemble'
    extraction_str = extraction_model.replace('/', '_').replace('.', '_')
    graph_creating_str = retrieval_model.replace('/', '_').replace('.', '_')
    if linking_model is None:
        linking_model = retrieval_model
    linking_str = f"{linking_model.replace('/', '_').replace('.', '_')}_linking_{linking}_top_k_{link_top_k}"
    if oracle_extraction:
        linking_str += '_oracle_ie'
    dpr_only_str = '_dpr_only' if dpr_only else ''
    run_output_path = f'exp/{dataset_name}_run_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}_{linking_str}{dpr_only_str}.json'

    pytrec_metrics = {'map', 'ndcg'}
    metrics = defaultdict(float)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_metrics)
    if os.path.isfile(run_output_path):
        run_dict = json.load(open(run_output_path))
        print(f'Log file found at {run_output_path}, len: {len(run_dict["retrieved"])}')
    else:
        run_dict = {'retrieved': {}, 'log': {}}  # for pytrec_eval

    to_update_run = False
    for i, query in tqdm(enumerate(dataset), total=len(dataset), desc='Evaluating queries'):
        query_text = query['text']
        query_id = query['id']
        if query_id in run_dict['retrieved']:
            continue
        supporting_docs = query['paragraphs']
        if oracle_extraction:
            oracle_triples = []
            for p in supporting_docs:
                oracle_triples += hipporag.get_facts_by_corpus_idx(hipporag.get_corpus_idx_by_passage_idx(p['idx']))[0]
        else:
            oracle_triples = None
        ranks, scores, log = hipporag.rank_docs(query_text, doc_top_k=10, link_top_k=link_top_k, linking=linking, oracle_triples=oracle_triples)

        retrieved_docs = [corpus[r] for r in ranks]
        to_update_run = True

        if linking in ['ner_to_node', 'query_to_node', 'query_to_fact']:  # evaluate the recall of nodes from supporting documents
            # get oracle nodes
            if oracle_triples is None:
                oracle_triples = []
                for p in supporting_docs:
                    oracle_triples += hipporag.get_facts_by_corpus_idx(hipporag.get_corpus_idx_by_passage_idx(p['idx']))[0]
            oracle_nodes = set([t[0] for t in oracle_triples]).union(set([t[2] for t in oracle_triples]))

            # get linked nodes
            linked_nodes = set()
            if log is not None and len(log):
                log['nodes_in_supporting_doc'] = list(oracle_nodes)
                for item in log['linked_node_scores']:
                    assert isinstance(item, list) or isinstance(item, str)
                    if isinstance(item, list):  # item: mention -> node phrase
                        linked_nodes.add(item[1])
                    elif isinstance(item, str):
                        linked_nodes.add(item)
                    if link_top_k is not None and len(linked_nodes) >= link_top_k:
                        break

                # calculate recall
                node_precision = len(oracle_nodes.intersection(set(linked_nodes))) / len(linked_nodes) if len(linked_nodes) > 0 else 0
                node_recall = len(oracle_nodes.intersection(set(linked_nodes))) / len(oracle_nodes) if len(oracle_nodes) > 0 else 0
                node_hit = True if len(oracle_nodes.intersection(set(linked_nodes))) > 0 else False
                metrics['node_precision'] += node_precision
                metrics['node_recall'] += node_recall
                metrics['node_hit'] += node_hit
            else:
                hipporag.logger.info(f'No linked nodes found for query {query_id}.')

        log['query'] = query_text
        run_dict['retrieved'][query_id] = {doc['idx']: score for doc, score in zip(retrieved_docs, scores)}
        run_dict['log'][query_id] = log
    # end each query

    if to_update_run:
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

    if detail:
        logs = detailed_log(dataset, run_dict, eval_res, chunk, dpr_only=dpr_only)
        detailed_log_output_path = f'exp/{dataset_name}_log_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}{linking_str}{dpr_only_str}.json'
        with open(detailed_log_output_path, 'w') as f:
            json.dump(logs, f)
        print(f'Detailed log saved to {detailed_log_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `sci_fact_test`, `fiqa_dev`.')
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--retrieval_model', type=str, help="Graph creating retriever name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--linking_model', type=str, help="Node linking model name, e.g., 'facebook/contriever', 'colbertv2'")
    parser.add_argument('--linking', type=str, choices=['ner_to_node', 'query_to_node', 'query_to_fact'])
    parser.add_argument('--doc_ensemble', action='store_true')
    parser.add_argument('--dpr_only', action='store_true')
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--oracle_ie', action='store_true')
    args = parser.parse_args()

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
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only,
                        linking_retriever_name=args.linking_model)

    with open(f'data/{args.dataset}.json') as f:
        dataset = json.load(f)

    if not args.dpr_only:
        link_top_k_list = [1, 2, 3, 5, 10, 20, 30]
        if args.linking == 'ner_to_node':
            link_top_k_list.append(None)
        for link_top_k in link_top_k_list:
            run_retrieve_beir(args.dataset, args.extraction_model, args.retrieval_model, args.linking_model, args.linking,
                              args.doc_ensemble, args.dpr_only, args.chunk, args.detail, link_top_k=link_top_k, oracle_extraction=args.oracle_ie)
    else:  # DPR only
        run_retrieve_beir(args.dataset, args.extraction_model, args.retrieval_model, args.linking_model, args.linking,
                          args.doc_ensemble, args.dpr_only, args.chunk, args.detail, link_top_k=1, oracle_extraction=args.oracle_ie)
