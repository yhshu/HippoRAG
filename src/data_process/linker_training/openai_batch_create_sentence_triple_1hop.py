import argparse
import json
import os
import pickle

from src.data_process.linker_training.openai_batch_create_sentence_triple import sentence_triple_response_format
from src.data_process.util import generate_hash
from src.lm_wrapper.util import openai_batch_create_api

sentence_triple_system_message = """Given a query, its answer and its relevant passage, where the passage contains some facts that answers this query. Your task is:
1. Extract a subsequence from the passage that helps answering this query.
2. Extract 1-2 triples (subject, predicate, object) that do help reasoning or answering the query from the piece you just extracted. Don't generate redundant irrelevant triples. 
3. Return the sentence and triple lists in JSON format."""

sentence_triple_demo_input = """Query: the capital of France
Answer: ["Paris"]
Passage: The Eiffel Tower is located in Paris and was constructed in 1887. Paris is an ancient city, one of the largest cities in the West, and the capital of France."""

sentence_triple_demo_output = """{
    "sentences": [
        "Paris is the capital of France.",
    ],
    "triples": [
        [
            "Paris",
            "capital of",
            "France"
        ]
    ]
}"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--dir', type=str, default='data/linker_training/queries')
    parser.add_argument('--datasets', nargs='+', type=str, help='A list of datasets, e.g., msmacro')
    args = parser.parse_args()

    cache_path = 'data/linker_training/sentence_triple_cache.pkl'
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, 'rb'))
    else:
        cache = {}  # custom_id -> {sentence, triples}

    jsonl_contents = []
    selected_datasets = args.datasets

    file_names = os.listdir(args.dir)
    for path in file_names:
        file_path = os.path.join(args.dir, path)
        if not os.path.isfile(file_path):
            continue
        data = json.load(open(os.path.join(args.dir, path), 'r'))

        dataset_label = path.split('.')[0]
        if selected_datasets is not None and not any(dataset in dataset_label for dataset in selected_datasets):
            continue

        for sample in data:
            for candidate in sample['candidates']:
                custom_id = f"{dataset_label}_{sample['id']}_{generate_hash(candidate['sentence'])}"
                if custom_id in cache:
                    continue

                user_prompt = f"Query: {sample['question']}\nAnswer: {json.dumps(sample['answers'])}\nPassage: {candidate['sentence']}"
                messages = [{'role': 'system', 'content': sentence_triple_system_message},
                            {'role': 'user', 'content': sentence_triple_demo_input},
                            {'role': 'assistant', 'content': sentence_triple_demo_output},
                            {'role': 'user', 'content': user_prompt}]
                jsonl_contents.append(json.dumps(
                    {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": args.model, "messages": messages, "max_tokens": 1024, "response_format": sentence_triple_response_format}}))

    corpus_jsonl_path = f'data/linker_training/sentence_triple_1hop_{args.model}_submission_{len(jsonl_contents)}.jsonl'
    openai_batch_create_api(corpus_jsonl_path, jsonl_contents)
