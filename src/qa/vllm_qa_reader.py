import sys

sys.path.append('.')

from src.qa.qa_reader import get_qa_input_messages, get_retrieved_items, evaluate_answer
from src.langchain_util import init_llm_client
from src.baselines.ircot import parse_prompt

import os
import argparse
import json
from tqdm import tqdm


def vllm_qa_read(data, demos, args, client, total_metrics, json_mode=False):
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

        all_messages.append(get_qa_input_messages(demos, retrieved, query, json_mode))

    if 'meta-llama/Llama-3' in client.llm_engine.model_config.served_model_name or 'Meta-Llama-3.' in client.llm_engine.model_config.served_model_name:
        from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt
        all_prompts = [langchain_message_to_llama_3_prompt(qa_message) for qa_message in all_messages]
    else:
        raise NotImplementedError("Please add the conversion for this model")

    print('QA prompt example:', all_prompts[0])

    from vllm import SamplingParams
    guided_options_request = None
    if json_mode:
        from src.util.llama_cpp_service import PROMPT_JSON_TEMPLATE
        guided_options_request = vllm.model_executor.guided_decoding.guided_fields.GuidedDecodingRequest(
            guided_json=PROMPT_JSON_TEMPLATE['qa_cot'])
    exp_label = '' if args.exp is None else f', exp: {args.exp}'
    print(f'Running QA on {args.dataset} dataset with {len(all_prompts)} samples{exp_label}...')
    vllm_output = client.generate(
        all_prompts,
        sampling_params=SamplingParams(max_tokens=512, temperature=0),
        guided_options_request=guided_options_request
    )
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    all_total_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

    assert len(all_responses) == len(data)
    for idx, sample in enumerate(data):
        pred_ans, em, f1, precision, recall = evaluate_answer(all_responses[idx], sample, json_mode)
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
    args.force_retry = True

    llm_model_name_processed = args.llm_model.replace('/', '_')
    retriever_name = args.retriever.replace('/', '_') if args.retriever else 'none'
    assert args.dataset is not None and len(args.dataset.strip()) > 0
    os.makedirs(f'exp/qa/{args.dataset}', exist_ok=True)
    exp_label = '' if args.exp is None else f'_{args.exp}'
    output_path = f'exp/qa/{args.dataset}/{retriever_name}_{llm_model_name_processed}_demo_{args.num_demo}_doc_{args.num_doc}{exp_label}.json'

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
        prompt_path = f'data/ircot_prompts/{prompt_dataset}/gold_with_3_distractors_context_cot_qa_codex.txt'
        corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
        demos = parse_prompt(prompt_path)

    total_metrics = {'qa_em': 0, 'qa_f1': 0, 'qa_precision': 0, 'qa_recall': 0}

    assert data and len(data)
    demos = demos[:args.num_demo]
    client = init_llm_client('vllm', args.llm_model, num_gpus=args.num_gpus)
    vllm_qa_read(data, demos, args, client, total_metrics)

    metric_str = ' '.join([f'{key}: {total_metrics[key] / len(data):.4f}' for key in total_metrics.keys()])
    print(metric_str)

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print('QA results saved to', output_path)
