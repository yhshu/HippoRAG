import sys

sys.path.append('.')

from src.qa.qa_reader import get_qa_input_messages, get_retrieved_items, evaluate_answer
from src.langchain_util import init_langchain_model
from src.baselines.ircot import parse_prompt

import os
import argparse
import json
from tqdm import tqdm


def remove_newlines_after_first(s):
    first_newline_pos = s.find('\n')
    if first_newline_pos == -1:
        return s
    part_before_first_newline = s[:first_newline_pos + 1]
    part_after_first_newline = s[first_newline_pos + 1:].replace('\n', '')
    return part_before_first_newline + part_after_first_newline


cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')


def vllm_qa_read(data, demos, args, client, total_metrics, processed_id_set):
    import vllm
    assert isinstance(client, vllm.LLM)
    all_messages = []
    for sample_idx, sample in tqdm(enumerate(data), desc='Creating prompts', total=len(data)):
        query = sample['question']
        if '_id' in sample or 'id' in sample:
            sample_id = sample['_id'] if '_id' in sample else sample['id']
        else:
            sample_id = sample_idx
        retrieved = get_retrieved_items(sample, sample_id, args.num_doc, args.dataset)
        sample['retrieved'] = retrieved

        all_messages.append(get_qa_input_messages(demos, retrieved, query))

    if 'meta-llama/Llama-3' in client.llm_engine.model_config.served_model_name:
        from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt
        all_prompts = [langchain_message_to_llama_3_prompt(qa_message) for qa_message in all_messages]
    else:
        raise NotImplementedError("Please add the conversion for this model")

    print('QA prompt example:', all_prompts[0])

    from vllm import SamplingParams

    vllm_output = client.generate(
        all_prompts,
        sampling_params=SamplingParams(max_tokens=512, temperature=0)
    )
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    all_total_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

    assert len(all_responses) == len(data)
    for idx, sample in enumerate(data):
        pred_ans, em, f1, precision, recall = evaluate_answer(all_responses[idx], sample)
        processed_id_set.add(sample['_id'] if '_id' in sample else sample['id'])
        sample['prediction'] = pred_ans
        sample['qa_em'] = em
        sample['qa_f1'] = f1
        sample['qa_precision'] = precision
        sample['qa_recall'] = recall
        total_metrics['qa_em'] += em
        total_metrics['qa_f1'] += f1
        total_metrics['qa_precision'] += precision
        total_metrics['qa_recall'] += recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='retrieval results or QA reading results', required=True)
    parser.add_argument('--exp', type=str, help='The experimental name', required=True)
    parser.add_argument('--data', type=str, help='retrieval results or QA reading results')
    parser.add_argument('--retriever', type=str, help='retriever name to distinguish different experiments')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='Specific model name')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of few-shot examples')
    parser.add_argument('--num_doc', type=int, default=5, help='the number of in-context documents')
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--force_retry', action='store_true')
    args = parser.parse_args()

    llm_model_name_processed = args.llm_model.replace('/', '_')
    retriever_name = args.retriever.replace('/', '_') if args.retriever else 'none'
    assert args.dataset is not None and len(args.dataset.strip()) > 0
    os.makedirs('exp/qa/{args.dataset}', exist_ok=True)
    exp_label = '' if args.exp is None else f'_{args.exp}'
    output_path = f'exp/qa/{args.dataset}/{retriever_name}_{llm_model_name_processed}_demo_{args.num_demo}_doc_{args.num_doc}{exp_label}.json'

    processed_id_set = set()
    total_metrics = {'qa_em': 0, 'qa_f1': 0, 'qa_precision': 0, 'qa_recall': 0}
    if args.data:
        data = json.load(open(args.data, 'r'))
    else:
        data = json.load(open(f'data/{args.dataset}.json', 'r'))
        print('Dataset without retrieval results is loaded')

    if args.retriever == 'none':
        args.num_doc = 0

    prompt_dataset = args.dataset if args.dataset in ['musique', '2wikimultihopqa', 'hotpotqa'] else 'musique'
    if args.num_doc == 0:
        prompt_path = f'data/ircot_prompts/{prompt_dataset}/no_context_cot_qa_codex.txt'
        data = json.load(open(f'data/{args.dataset}.json', 'r'))
        demos = parse_prompt(prompt_path, False)
    else:
        if args.force_retry is False:
            if os.path.isfile(output_path):  # resume from previous results
                data = json.load(open(output_path, 'r'))
                for key in total_metrics.keys():
                    total_metrics[key] = sum([sample[key] for sample in data if key in sample])
        prompt_path = f'data/ircot_prompts/{prompt_dataset}/gold_with_3_distractors_context_cot_qa_codex.txt'
        corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
        demos = parse_prompt(prompt_path)

    # processed id set
    if args.force_retry is False:
        if args.dataset in ['hotpotqa', '2wikimultihopqa']:
            processed_id_set = {sample['_id'] for sample in data if 'prediction' in sample}
        elif args.dataset in ['musique']:
            processed_id_set = {sample['id'] for sample in data if 'prediction' in sample}
    else:
        processed_id_set = set()
        total_metrics = {'qa_em': 0, 'qa_f1': 0, 'qa_precision': 0, 'qa_recall': 0}

    assert data and len(data)
    demos = demos[:args.num_demo]
    client = init_langchain_model('vllm', args.llm_model, num_gpus=args.num_gpus)
    vllm_qa_read(data, demos, args, client, total_metrics, processed_id_set)

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print('QA results saved to', output_path)

    metric_str = ' '.join([f'{key}: {total_metrics[key] / len(data):.4f}' for key in total_metrics.keys()])
    print(metric_str)
