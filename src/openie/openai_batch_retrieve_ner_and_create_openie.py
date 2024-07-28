import argparse
import json

from openai import OpenAI
from tqdm import tqdm

from src.langchain_util import num_tokens_by_tiktoken
from src.openie_extraction_instructions import openie_post_ner_instruction, openie_post_ner_input_one_shot, \
    openie_post_ner_output_one_shot, openie_post_ner_frame
from src.openie_with_retrieval_option_parallel import load_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_id', type=str, help="OpenAI file ID to retrieve", required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Max tokens per prompt')
    parser.add_argument('--num_passages', type=str, default='all')
    args = parser.parse_args()

    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(args.dataset, args.model_name, args.num_passages, True)
    passage_dict = {str(p['idx']): p['passage'] for p in retrieval_corpus}  # custom_id to passage

    client = OpenAI(max_retries=5, timeout=60)
    content = client.files.content(file_id=args.file_id)

    # save file to do OpenIE after NER
    lines = content.text.strip().split("\n")
    print('content length:', len(lines))

    # preparing OpenIE submission to OpenAI
    corpus_jsonl_path = f'output/openai_batch_submission_openie_{args.dataset}_{args.model_name}.jsonl'
    jsonl_contents = []
    total_tokens = 0
    for line in tqdm(lines):
        response = json.loads(line)
        if response['error'] is None:
            content = response['response']['body']['choices'][0]['message']['content']
        try:
            ner = json.loads(content)
        except Exception as e:
            print('Loading NER json exception:', e)
            print('Content:', content)
            continue

        custom_id = response['custom_id']
        passage = passage_dict[custom_id]
        user_message = openie_post_ner_frame.replace("{passage}", passage).replace("{named_entity_json}", json.dumps(ner))
        openie_messages = [{'role': 'system', 'content': openie_post_ner_instruction},
                           {'role': 'user', 'content': openie_post_ner_input_one_shot},
                           {'role': 'assistant', 'content': openie_post_ner_output_one_shot},
                           {'role': 'user', 'content': user_message}]

        total_tokens += num_tokens_by_tiktoken(str(openie_messages))
        openie_submission = json.dumps(
            {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": args.model_name, "messages": openie_messages,
                      "max_tokens": args.max_tokens, "response_format": {"type": "json_object"}}})
        jsonl_contents.append(openie_submission)

    print("Total prompt tokens:", total_tokens)
    print("Approximate costs for prompt tokens using GPT-4o-mini Batch API:", round(0.075 * total_tokens / 1e6, 3))
    print("Approximate costs for prompt tokens using GPT-3.5-turbo-0125 Batch API", round(0.25 * total_tokens / 1e6, 3))
    print("Approximate costs for prompt tokens using GPT-4o Batch API:", round(2.5 * total_tokens / 1e6, 3))

    # Save to the batch file
    with open(corpus_jsonl_path, 'w') as f:
        f.write('\n'.join(jsonl_contents))
        print("Batch file saved to", corpus_jsonl_path, 'len: ', len(jsonl_contents))

    batch_input_file = client.files.create(file=open(corpus_jsonl_path, 'rb'), purpose='batch')
    batch_obj = client.batches.create(input_file_id=batch_input_file.id, endpoint='/v1/chat/completions',
                                      completion_window='24h',
                                      metadata={'description': f"HippoRAG OpenIE for {args.dataset}, len: {len(jsonl_contents)}"})
    print(batch_obj)
    print("Go to https://platform.openai.com/batches/ or use OpenAI file API to get the output file ID after the batch job is done.")
