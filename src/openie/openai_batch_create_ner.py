import argparse
import json

from src.openie_extraction_instructions import ner_prompts
from src.openie_with_retrieval_option_parallel import load_corpus


def named_entity_recognition_for_corpus_openai_batch(dataset_name: str,  num_passages, model_name: str):
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(dataset_name, model_name, num_passages, True)

    # output corpus to a file to upload to OpenAI
    corpus_jsonl_path = f'output/batch_ner_{arg_str}.jsonl'
    jsonl_contents = []
    for i, passage in enumerate(retrieval_corpus):
        ner_messages = ner_prompts.format_prompt(user_input=passage)

        jsonl_contents.append(json.dumps({"custom_id": i, "method": "POST", "url": "/v1/chat/completions", "messages": ner_messages,
                                          "max_tokens": 4096}))
    with open(corpus_jsonl_path, 'w') as f:
        f.write('\n'.join(jsonl_contents))

    from openai import OpenAI
    client = OpenAI(max_retries=5, timeout=60)
    batch_input_file = client.files.create(file=open(corpus_jsonl_path, 'rb'), purpose='batch')
    batch_obj = client.batches.create(input_file_id=batch_input_file.id, endpoint='/v1/chat/completions', completion_window='24h',
                                      metadata={'description': f"HippoRAG OpenIE for {dataset_name}"})
    print(batch_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_passages', type=str, default='10')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='Specific model name')
    args = parser.parse_args()

    named_entity_recognition_for_corpus_openai_batch(args.dataset, args.num_passages, args.model_name)
