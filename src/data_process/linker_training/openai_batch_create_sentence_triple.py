import sys

from src.lm_wrapper.util import openai_batch_create_api

sys.path.append('.')

import argparse
import json
import os
import pickle

sentence_triple_system_message = """Given a query and its supporting passages for this query, each passage contains some facts that is relevant to this query. Your task is:
1. Extract a subsequence from EACH passage that helps reasoning or answering this query. The number of subsequences should be the same as the number of passages.
2. Extract 1-2 triples (subject, predicate, object) that do help reasoning or answering the query from EACH of the piece you just extracted. Don't generate redundant irrelevant triples.

Requirements:

- The query may require multi-hop reasoning, so you may need to consolidate information across passages.
- The triples you provide should form a inferential chain for the query. 
- Return the sentence and triple lists in JSON format."""

sentence_triple_demo_input = """Query: What is the capital of the country where the Eiffel Tower is located?

Passage 1:
The Eiffel Tower is located in Paris and was constructed in 1887. 

Passage 2:
Paris is an ancient city, one of the largest cities in the West, and the capital of France."""

sentence_triple_demo_output = """{
    "sentences": [
        "The Eiffel Tower is located in Paris.",
        "Paris is the capital of France."
    ],
    "triples": [
        [
            "Eiffel Tower",
            "located in",
            "Paris"
        ],
        [
            "Paris",
            "capital of",
            "France"
        ]
    ]
}"""

sentence_triple_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentences_and_triples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentences": {"type": "array", "items": {"type": "string"}},
                "triples": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            },
            "required": ["sentences", "triples"],
            "additionalProperties": False,
        }
    }
}

decomposition_to_triple_system_message = ("Given a query and its inferential chain (2-4 steps), your task is to convert this chain into a set of triples "
                                          "(subject, predicate, object) that represent the reasoning process. Return triples in JSON format.")

decomposition_to_triple_demo_input = """Query: What is the capital of the country where the Eiffel Tower is located?

Step 1: {"question": "Eiffel Tower >> located country", "answer": "France"}
Step 2: {"question": "What is the capital of #1", "answer": "Paris"}"""

decomposition_to_triple_demo_output = """{
    "triples": [
        ["Eiffel Tower", "located country", "France"], 
        ["Paris", "capital of", "France"]
    ]
}"""

triple_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "triples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triples": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            },
            "required": ["triples"],
            "additionalProperties": False,
        }
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--dir', type=str, default='data/linker_training/queries')
    parser.add_argument('--datasets', nargs='+', type=str, help='A list of datasets')
    args = parser.parse_args()

    cache_path = 'data/linker_training/sentence_triple_cache.pkl'
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, 'rb'))
    else:
        cache = {}  # custom_id -> {sentence, triples}

    jsonl_contents = []

    for path in os.listdir(args.dir):
        if not os.path.isfile(os.path.join(args.dir, path)):
            continue
        if args.datasets is not None and not any(dataset in path for dataset in args.datasets):
            continue

        data = json.load(open(os.path.join(args.dir, path), 'r'))
        dataset_label = path.split('.')[0]
        for sample in data:
            custom_id = f"{dataset_label}_{sample['id']}"
            if custom_id in cache:
                continue

            if 'question_decomposition' in sample:
                user_prompt = f"Query: {sample['question']}\n"
                for i, step in enumerate(sample['question_decomposition']):
                    this_step = {}
                    this_step['question'] = step['question']
                    this_step['answer'] = step['answer']
                    user_prompt += f"\nStep {i + 1}: {json.dumps(this_step)}"
                messages = [{'role': 'system', 'content': decomposition_to_triple_system_message},
                            {'role': 'user', 'content': decomposition_to_triple_demo_input},
                            {'role': 'assistant', 'content': decomposition_to_triple_demo_output},
                            {'role': 'user', 'content': user_prompt}]
                jsonl_contents.append(json.dumps(
                    {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": args.model, "messages": messages, "max_tokens": 1024, "response_format": triple_response_format}}))
            else:
                relevant_passages = []
                for candidate in sample['candidates']:
                    relevant_passages.append(candidate['sentence'])

                user_prompt = f"Query: {sample['question']}"
                for i, passage in enumerate(relevant_passages):
                    user_prompt += f"\n\nPassage {i + 1}:\n{passage}"
                messages = [{'role': 'system', 'content': sentence_triple_system_message},
                            {'role': 'user', 'content': sentence_triple_demo_input},
                            {'role': 'assistant', 'content': sentence_triple_demo_output},
                            {'role': 'user', 'content': user_prompt}]
                jsonl_contents.append(json.dumps(
                    {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                     "body": {"model": args.model, "messages": messages, "max_tokens": 1024, "response_format": sentence_triple_response_format}}))

    corpus_jsonl_path = f'data/linker_training/sentence_triple_{args.model}_submission_{len(jsonl_contents)}.jsonl'
    openai_batch_create_api(corpus_jsonl_path, jsonl_contents)
