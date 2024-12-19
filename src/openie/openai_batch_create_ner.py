import sys
sys.path.append('.')
import argparse
import json
from math import ceil
from src.langchain_util import num_tokens_by_tiktoken
from src.openie_extraction_instructions import ner_output_one_shot, ner_input_one_shot, ner_instruction
from src.openie_with_retrieval_option_parallel import load_corpus

def check_duplicate_custom_id(jsonl_contents):
    custom_id_set = set()
    duplication = False
    for item in jsonl_contents:
        item = json.loads(item)
        custom_id = item['custom_id']
        if custom_id not in custom_id_set:
            custom_id_set.add(custom_id)
        else:
            duplication = True
    if duplication:
        print(f'The number of unique custom ids {len(custom_id_set)} for a list with {len(jsonl_contents)} elements')
    return duplication

def split_into_batches(jsonl_contents, batch_size):
    """
    Split the input JSONL content into multiple batches.
    :param jsonl_contents: List of JSON strings.
    :param batch_size: Maximum number of samples per batch.
    :return: List of batches, where each batch is a list of JSON strings.
    """
    total_batches = ceil(len(jsonl_contents) / batch_size)
    batches = [jsonl_contents[i * batch_size: (i + 1) * batch_size] for i in range(total_batches)]
    return batches

def save_and_submit_batch(client, batch_contents, corpus_jsonl_path_template, dataset_name, model_name):
    """
    Save each batch to a file and submit it using the OpenAI Batch API.
    :param client: OpenAI client.
    :param batch_contents: List of JSONL contents for a single batch.
    :param corpus_jsonl_path_template: Path template for saving JSONL files.
    :param dataset_name: Name of the dataset.
    :param model_name: Model name used for processing.
    """
    batch_index = 0
    for batch in batch_contents:
        batch_path = corpus_jsonl_path_template.format(batch_index)
        with open(batch_path, 'w') as f:
            f.write('\n'.join(batch))
        print(f"Batch file saved to {batch_path}, length: {len(batch)}")

        # Call OpenAI Batch API for each batch
        batch_input_file = client.files.create(file=open(batch_path, 'rb'), purpose='batch')
        batch_obj = client.batches.create(
            input_file_id=batch_input_file.id, endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={'description': f"HippoRAG OpenIE Batch {batch_index} for {dataset_name}, len: {len(batch)}"}
        )
        print(batch_obj)
        print(f"Batch {batch_index} submitted successfully.")
        batch_index += 1

def named_entity_recognition_for_corpus_openai_batch(dataset_name: str, num_passages, model_name: str, max_tokens=4096, batch_size=50000):
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(dataset_name, model_name, num_passages, True)

    # Output corpus to a file to upload to OpenAI
    corpus_jsonl_path_template = f'output/ner_batch_{dataset_name[1:]}_{model_name}_batch_{{}}.jsonl'
    jsonl_contents = []
    total_tokens = 0

    for idx, passage in enumerate(retrieval_corpus):
        ner_messages = [{'role': 'system', 'content': ner_instruction},
                        {'role': 'user', 'content': ner_input_one_shot},
                        {'role': 'assistant', 'content': ner_output_one_shot},
                        {'role': 'user', 'content': f"Paragraph:```\n{passage['passage']}\n```"}]
        total_tokens += num_tokens_by_tiktoken(str(ner_messages))

        # Custom_id must be string
        jsonl_contents.append(json.dumps(
            {"custom_id": str(idx), "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": model_name, "messages": ner_messages,
                      "max_tokens": max_tokens, "response_format": {"type": "json_object"}}}))

    assert check_duplicate_custom_id(jsonl_contents) is False, "Duplicate custom ids"
    print("Total prompt tokens:", total_tokens)
    print("Approximate costs for prompt tokens using GPT-4o-mini Batch API:", round(0.075 * total_tokens / 1e6, 3))
    print("Approximate costs for prompt tokens using GPT-3.5-turbo-0125 Batch API", round(0.25 * total_tokens / 1e6, 3))
    print("Approximate costs for prompt tokens using GPT-4o Batch API:", round(1.25 * total_tokens / 1e6, 3))

    # Split into batches
    batches = split_into_batches(jsonl_contents, batch_size)
    print(f"Number of batches: {len(batches)}")

    # Call OpenAI Batch API for each batch
    from openai import OpenAI
    client = OpenAI(max_retries=5, timeout=60)
    save_and_submit_batch(client, batches, corpus_jsonl_path_template, dataset_name, model_name)
    print("All batches submitted successfully. Go to https://platform.openai.com/batches/ or use OpenAI file API to get the output file ID after the batch job is done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--batch_size', type=int, default=50000, help='Number of samples per batch')
    args = parser.parse_args()

    named_entity_recognition_for_corpus_openai_batch(args.dataset, args.num_passages, args.model_name, batch_size=args.batch_size)
