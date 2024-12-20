import sys
sys.path.append('.')

import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm
from transformers.hf_argparser import string_to_bool
from src.langchain_util import num_tokens_by_tiktoken
from src.openie_extraction_instructions import (
    openie_post_ner_instruction, openie_post_ner_input_one_shot,
    openie_post_ner_output_one_shot, openie_post_ner_frame
)
from src.openie_with_retrieval_option_parallel import load_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Max tokens per prompt')
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--skip_openie', type=str, default='f')
    parser.add_argument('--file_ids', type=str, help="List of OpenAI file IDs to retrieve", nargs='+', required=True)
    args = parser.parse_args()

    print("Retrieving OpenAI file ID list:", args.file_ids)

    skip_openie = string_to_bool(args.skip_openie)
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(
        args.dataset, args.model_name, args.num_passages, True
    )
    passage_dict = {
        str(i): p['passage']
        for i, p in enumerate(retrieval_corpus)
    }  # custom_id to passage

    client = OpenAI(max_retries=5, timeout=60)

    output_dir = "output/openie_batches"
    os.makedirs(output_dir, exist_ok=True)

    for file_idx, openai_file_id in enumerate(args.file_ids):  # Loop over each file ID
        print(f"Processing file ID: {openai_file_id}")
        openie_submission_jsonl = []
        total_tokens = 0

        content = client.files.content(file_id=openai_file_id)
        lines = content.text.strip().split("\n")
        print(f'File ID {openai_file_id} - content length:', len(lines))

        for line in tqdm(lines, desc=f"Processing file {openai_file_id}"):
            response = json.loads(line)
            if response['error'] is None:
                content = response['response']['body']['choices'][0]['message']['content']
            try:
                ner = json.loads(content)
            except Exception as e:
                print('Loading NER json exception:', e)
                print('Content:', content)
                continue

            if not skip_openie:
                custom_id = f"global_{response['custom_id']}"  # Unique custom ID
                passage = passage_dict.get(response['custom_id'], "")
                user_message = openie_post_ner_frame.replace("{passage}", passage).replace("{named_entity_json}", json.dumps(ner))
                openie_messages = [
                    {'role': 'system', 'content': openie_post_ner_instruction},
                    {'role': 'user', 'content': openie_post_ner_input_one_shot},
                    {'role': 'assistant', 'content': openie_post_ner_output_one_shot},
                    {'role': 'user', 'content': user_message}
                ]

                total_tokens += num_tokens_by_tiktoken(str(openie_messages))
                openie_submission = json.dumps({
                    "custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": args.model_name, "messages": openie_messages,
                              "max_tokens": args.max_tokens, "response_format": {"type": "json_object"}}
                })
                openie_submission_jsonl.append(openie_submission)

        if not skip_openie and openie_submission_jsonl:
            openie_submission_file = os.path.join(output_dir, f"openie_{args.dataset}_{args.model_name}_batch_{file_idx}.jsonl")
            with open(openie_submission_file, 'w') as f:
                f.write('\n'.join(openie_submission_jsonl))
                print(f"Batch file saved to {openie_submission_file}, len: {len(openie_submission_jsonl)}")

            batch_input_file = client.files.create(file=open(openie_submission_file, 'rb'), purpose='batch')
            batch_obj = client.batches.create(
                input_file_id=batch_input_file.id, endpoint='/v1/chat/completions',
                completion_window='24h',
                metadata={'description': f"HippoRAG OpenIE for {args.dataset} file {file_idx}, len: {len(openie_submission_jsonl)}"}
            )
            print(batch_obj)

    print()
    print("Go to https://platform.openai.com/batches/ or use OpenAI file API to get the output file ID after the batch job is done.")
