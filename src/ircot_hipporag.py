import os
import sys

sys.path.append('.')

from typing import Union

from collections import defaultdict

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model
from transformers.hf_argparser import string_to_bool
import argparse
import json

from tqdm import tqdm

from hipporag import HippoRAG

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


def parse_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []

    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split('\n', 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        answer = qa_pair[1].strip()

        parsed_data.append({
            'metadata': metadata,
            'document': document_text,
            'question': question,
            'answer': answer
        })

    return parsed_data


def retrieve_step(query: str, corpus, top_k: int, hipporag: HippoRAG, dataset_name: str, link_top_k: Union[None, int], linking='ner_to_node', oracle_triples=None):
    ranks, scores, logs = hipporag.rank_docs(query, doc_top_k=top_k, link_top_k=link_top_k, linking=linking, oracle_triples=oracle_triples)
    retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    return retrieved_passages, scores, logs


def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """
    prompt_demo = ''
    for sample in few_shot:
        prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print(e)
        return ''
    return response_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--linker', type=str, default='facebook/contriever')
    parser.add_argument('--reranker', type=str)
    parser.add_argument('--linking', type=str, default='ner_to_node', help='linking method for the entry point of the graph')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--num_demo', type=int, default=1, help='the number of demo samples')
    parser.add_argument('--max_steps', type=int, required=True, default=1)
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    parser.add_argument('--link_top_k', type=int, help='the number of linked nodes at each retrieval step')
    parser.add_argument('--doc_ensemble', type=str, default='t')
    parser.add_argument('--dpr_only', action='store_true')
    parser.add_argument('--graph_alg', type=str, default='ppr')
    parser.add_argument('--graph_type', type=str, default='facts_and_sim')
    parser.add_argument('--wo_node_spec', action='store_true')
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    parser.add_argument('--recognition_threshold', type=float, default=0.9)
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--force_retry', action='store_true')
    parser.add_argument('--do_eval', type=str, default='t')
    parser.add_argument('--directed', action='store_true')
    args = parser.parse_args()

    # set langchain cache
    set_llm_cache(SQLiteCache(database_path=".ircot_hipporag.db"))

    do_eval = string_to_bool(args.do_eval)

    # check args
    if args.linking in ['query_to_node', 'query_to_fact', 'query_to_passage']:
        assert args.link_top_k, 'link_top_k should be provided for query_to_node or query_to_fact'

    # Please set environment variable OPENAI_API_KEY
    doc_ensemble = string_to_bool(args.doc_ensemble)

    llm_model_name_processed = args.llm_model.replace('/', '_').replace('.', '_')
    colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    hipporag = HippoRAG(args.dataset, extraction_model=args.llm, extractor_name=args.llm_model, graph_creating_retriever_name=args.retriever,
                        linker_name=args.linker,
                        doc_ensemble=doc_ensemble, node_specificity=not (args.wo_node_spec), sim_threshold=args.sim_threshold,
                        colbert_config=colbert_configs, dpr_only=args.dpr_only, graph_alg=args.graph_alg, damping=args.damping, recognition_threshold=args.recognition_threshold,
                        reranker_name=args.reranker, graph_type=args.graph_type, directed_graph=args.directed)

    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    max_steps = args.max_steps

    assert max_steps
    if max_steps > 1:
        if 'hotpotqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif 'musique' in args.dataset:
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif '2wikimultihopqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        else:
            prompt_path = f'data/ircot_prompts/{args.dataset}/gold_with_3_distractors_context_cot_qa_codex.txt'

        few_shot_samples = parse_prompt(prompt_path)[:args.num_demo]

    doc_ensemble_str = f'doc_ensemble_{args.recognition_threshold}' if doc_ensemble else 'no_ensemble'

    if args.dpr_only:
        dpr_only_str = 'dpr_only'
    else:
        dpr_only_str = 'hipporag'

    os.makedirs(f'output/ircot_retrieval/{args.dataset}', exist_ok=True)
    rerank_str = f'_RE_{args.reranker}' if args.reranker else ''
    graph_type_str = ''
    if 'passage_node' in args.graph_type:
        graph_type_str = '_GT_pn'
        if 'unidirectional' in args.graph_type:
            graph_type_str += 'u'

    output_path = (
        f'output/ircot_retrieval/{args.dataset}/{args.dataset}_{dpr_only_str}{graph_type_str}_E_{llm_model_name_processed}_R_{hipporag.graph_creating_retriever_name_processed}_L_{hipporag.linking_retriever_name_processed}_{args.linking}{rerank_str}'
        f'_demo_{args.num_demo}_step_{max_steps}_top_{args.top_k}_{args.graph_alg}_damp_{args.damping}_sim_{args.sim_threshold}')

    if args.wo_node_spec:
        output_path += 'wo_node_spec'
    if args.link_top_k:
        output_path += f'_LT_{args.link_top_k}'
    output_path += '.json'
    print('Log file will be saved to', output_path)

    k_list = [1, 2, 5, 10]
    metrics_sum = defaultdict(float)  # the sum of metrics for all samples

    force_retry = args.force_retry

    if force_retry:
        results = []
        processed_ids = set()
    else:
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            print(f'Loaded {len(results)} results from {output_path}')
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                processed_ids = {sample['_id'] for sample in results}
            else:
                processed_ids = {sample['id'] for sample in results}

            for sample in results:
                for k in k_list:
                    metrics_sum[f"recall_{k}"] += sample['recall'][str(k)]
                    metrics_sum[f"all_recall_{k}"] += sample['all_recall'][str(k)]
                    metrics_sum[f"any_recall_{k}"] += sample['any_recall'][str(k)]
                metrics_sum["node_precision"] += sample['node_precision']
                metrics_sum["node_recall"] += sample['node_recall']
                metrics_sum["node_hit"] += sample['node_hit']
        except Exception as e:
            print(e)
            print('Results file maybe empty, cannot be loaded.')
            results = []
            processed_ids = set()
            metrics_sum = defaultdict(float)  # the sum of metrics for all samples

    if len(results) > 0:
        for key in metrics_sum.keys():
            print(key, round(metrics_sum[key] / len(results), 4))
        print()

    for sample_idx, sample in tqdm(enumerate(data), total=len(data), desc='IRCoT retrieval'):  # for each sample
        if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample_id = sample['_id']
        elif 'id' in sample:
            sample_id = sample['id']
        else:
            sample_id = str(sample_idx)

        if sample_id in processed_ids:
            continue

        query = sample['question']
        logs_for_all_steps = {}

        gold_docs = []
        if args.dataset in ['2wikimultihopqa']:
            for item in sample['supporting_facts']:
                title = item[0]
                for c in sample['context']:
                    if c[0] == title:
                        gold_docs.append(c[0] + '\n' + ' '.join(c[1]))
                        break
        elif args.dataset in ['musique']:
            gold_docs = [item['title'] + '\n' + item['paragraph_text'] for item in sample['paragraphs'] if item['is_supporting']]

        oracle_triples = None
        if hipporag.reranker_name is not None and hipporag.reranker_name in ['oracle_triple']:
            assert len(gold_docs) > 0
            oracle_triples = []
            for p in gold_docs:
                assert len(p) > 0 and '\n' in p
                oracle_triples += hipporag.get_triples_and_triple_ids_by_passage_content(p)[0]
        retrieved_passages, scores, logs = retrieve_step(query, corpus, args.top_k, hipporag, args.dataset, args.link_top_k, args.linking, oracle_triples)

        it = 1
        logs_for_all_steps[it] = logs

        thoughts = []
        retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}

        while it < max_steps:  # for each iteration of IRCoT
            new_thought = reason_step(args.dataset, few_shot_samples, query, retrieved_passages[:args.top_k], thoughts, hipporag.client)
            thoughts.append(new_thought)
            if 'So the answer is:' in new_thought:
                break
            it += 1

            new_retrieved_passages, new_scores, logs = retrieve_step(new_thought, corpus, args.top_k, hipporag, args.dataset, args.link_top_k, args.linking, oracle_triples)
            logs_for_all_steps[it] = logs

            for passage, score in zip(new_retrieved_passages, new_scores):
                if passage in retrieved_passages_dict:
                    retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                else:
                    retrieved_passages_dict[passage] = score

            retrieved_passages, scores = zip(*retrieved_passages_dict.items())

            sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
            retrieved_passages, scores = zip(*sorted_passages_scores)
        # end iteration

        # calculate recall
        if do_eval:
            if args.dataset in ['hotpotqa', 'hotpotqa_train']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
            elif args.dataset in ['2wikimultihopqa']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
            else:
                gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
                gold_items = set([item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_passages])
                retrieved_items = retrieved_passages

            # record results
            phrases_in_gold_docs = []
            if not hipporag.dpr_only:
                for gold_passage in gold_passages:
                    if isinstance(gold_passage, dict):
                        passage_content = gold_passage['text'] if 'text' in gold_passage else gold_passage['paragraph_text']
                        phrases_in_gold_docs.append(hipporag.get_phrases_in_doc_by_str(passage_content))
                    # elif isinstance(gold_passage, list) and len(gold_passage) == 2 and isinstance(gold_passage[1], int):

            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                sample['supporting_docs'] = [item for item in sample['supporting_facts']]
            else:
                sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
                del sample['paragraphs']

            if len(phrases_in_gold_docs):
                sample['nodes_in_gold_doc'] = json.dumps(phrases_in_gold_docs)

        sample['retrieved'] = retrieved_passages[:10]
        sample['retrieved_scores'] = json.dumps(list(scores)[:10])

        logs_for_first_step = logs_for_all_steps[1]
        if logs_for_first_step is not None:
            for key in logs_for_first_step.keys():
                sample[key] = logs_for_first_step[key]
        sample['thoughts'] = thoughts

        # calculate node precision/recall/Hit
        if do_eval and logs_for_first_step is not None and not hipporag.dpr_only:
            linked_nodes = set()
            linked_node_scores = logs_for_first_step.get('linked_node_scores', '')
            if isinstance(linked_node_scores, str) and linked_node_scores.startswith('{') and linked_node_scores.endswith('}'):
                linked_node_scores = json.loads(linked_node_scores)
            for link in linked_node_scores:
                if isinstance(link, list):
                    linked_nodes.add(link[1])
                elif isinstance(link, str):
                    linked_nodes.add(link)
            oracle_nodes = set()
            for passage_phrases in phrases_in_gold_docs:
                for phrase in passage_phrases:
                    oracle_nodes.add(phrase)

            linked_nodes = set([node for node in linked_nodes if '\n' not in node])  # remove passage nodes
            node_precision = len(linked_nodes.intersection(oracle_nodes)) / len(linked_nodes) if len(linked_nodes) > 0 else 0.0
            node_recall = len(linked_nodes.intersection(oracle_nodes)) / len(oracle_nodes) if len(oracle_nodes) > 0 else 0.0
            node_hit = 1.0 if len(linked_nodes.intersection(oracle_nodes)) > 0 else 0.0
            sample['node_precision'] = node_precision
            sample['node_recall'] = node_recall
            sample['node_hit'] = node_hit
            metrics_sum['node_precision'] += node_precision
            metrics_sum['node_recall'] += node_recall
            metrics_sum['node_hit'] += node_hit

            if hipporag.reranker is not None:
                if 'rerank' in logs_for_first_step and 'facts_before_rerank' in logs_for_first_step['rerank']:
                    facts_before_rerank = logs_for_first_step['rerank']['facts_before_rerank']
                    nodes_before_rerank = set([fact[0] for fact in facts_before_rerank] + [fact[2] for fact in facts_before_rerank if len(fact) == 3])
                    fact_after_rerank = logs_for_first_step['rerank']['facts_after_rerank']
                    nodes_after_renank = set([fact[0] for fact in fact_after_rerank] + [fact[2] for fact in fact_after_rerank if len(fact) == 3])

                    node_precision_before_rerank = len(nodes_before_rerank.intersection(oracle_nodes)) / len(nodes_before_rerank) if len(nodes_before_rerank) > 0 else 0.0
                    node_recall_before_rerank = len(nodes_before_rerank.intersection(oracle_nodes)) / len(oracle_nodes) if len(oracle_nodes) > 0 else 0.0
                    node_precision_after_rerank = len(nodes_after_renank.intersection(oracle_nodes)) / len(nodes_after_renank) if len(nodes_after_renank) > 0 else 0.0
                    node_recall_after_rerank = len(nodes_after_renank.intersection(oracle_nodes)) / len(oracle_nodes) if len(oracle_nodes) > 0 else 0.0
                    sample['node_precision_before_rerank'] = node_precision_before_rerank
                    sample['node_recall_before_rerank'] = node_recall_before_rerank
                    metrics_sum['node_precision_before_rerank'] += node_precision_before_rerank
                    metrics_sum['node_recall_before_rerank'] += node_recall_before_rerank

        # calculate passage retrieval recall
        if do_eval:
            recall = dict()
            print(f'idx: {sample_idx + 1} ', end='')
            for k in k_list:
                recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
                metrics_sum[f"recall_{k}"] += recall[k]

            # calculate all-recall: whether all gold items are retrieved
            all_recall_dict = dict()
            any_recall_dict = dict()
            for k in k_list:
                all_recall = 1 if all(t in retrieved_items[:k] for t in gold_items) else 0
                any_recall = 1 if any(t in retrieved_items[:k] for t in gold_items) else 0
                metrics_sum[f'all_recall_{k}'] += all_recall
                metrics_sum[f'any_recall_{k}'] += any_recall
                if k in [1, 2, 5, 10]:
                    all_recall_dict[str(k)] = all_recall
                    any_recall_dict[str(k)] = any_recall

            sample['recall'] = recall
            sample['all_recall'] = all_recall_dict
            sample['any_recall'] = any_recall_dict

            for key in sorted(metrics_sum.keys(), key=lambda x: (len(x), x)):
                num = round(metrics_sum[key] / (sample_idx + 1), 4)
                assert 0.0 <= num <= 1.0
                print(f'{key} {num} ', end='')
            print()
            if max_steps > 1:
                print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

        results.append(sample)
        if (sample_idx + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)
    # end for each sample

    if do_eval:
        # show metrics
        for key in sorted(metrics_sum.keys(), key=lambda x: (len(x), x)):
            metrics_sum[key] /= len(data)
            print(key, round(metrics_sum[key], 3))
        if hipporag.reranker is not None:
            print('num_dpr', hipporag.statistics['num_dpr'])

    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved {len(results)} results to {output_path}')
