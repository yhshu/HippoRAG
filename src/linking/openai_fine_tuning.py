import argparse
import json
import os
import pickle
import random

from gritlm import GritLM
from openai import OpenAI
from tqdm import tqdm

from src.langchain_util import num_tokens_by_tiktoken
from src.pangu.retrieval_api import GritLMRetriever

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt-4o-mini')
    parser.add_argument("--suffix", type=str, default='musique_fact_linker')
    parser.add_argument("--upload", action='store_true')
    args = parser.parse_args()

    training_output_path = 'exp/fact_linker/openai_fact_linker_training_data.jsonl'
    validation_output_path = 'exp/fact_linker/openai_fact_linker_validation_data.jsonl'
    cache = pickle.load(open('data/linker_training/sentence_triple_cache.pkl', 'rb'))

    training = []
    validation = []
    test = []

    gritlm_model = GritLM('GritLM/GritLM-7B', torch_dtype='auto')
    retriever_instruction = 'Given a query, retrieve relevant facts that contribute to answering this query.'
    for file_name in os.listdir('data/linker_training/queries'):
        data = json.load(open(f'data/linker_training/queries/{file_name}', 'r'))
        data = random.sample(data, min(2500, len(data)))
        dataset_label = file_name.split('.')[0]
        if 'train' in dataset_label:
            split = 'train'
        elif 'valid' in dataset_label or 'dev' in dataset_label:
            split = 'validation'
        else:
            split = 'test'

        all_triples = []
        for sample in data:
            custom_id = f"{dataset_label}_{sample['id']}"
            if custom_id in cache:
                all_triples.extend(cache[custom_id]['triples'])

        print(f"Loaded {len(all_triples)} facts from {file_name}")

        retriever = GritLMRetriever([json.dumps(triple) for triple in all_triples], model=gritlm_model, instruction=retriever_instruction)
        for sample in tqdm(data):
            query = sample['question']
            labels = []  # facts to link to

            custom_id = f"{dataset_label}_{sample['id']}"
            if custom_id in cache:
                labels = cache[custom_id]['triples']

            retrieved = retriever.get_top_k_sentences(query, k=30)
            retrieved = [eval(triple) for triple in retrieved]
            # check if labels are in retrieved facts
            for label in labels:
                if label not in retrieved:
                    retrieved = [label] + retrieved[:-1]
            # order retrieved facts by subject
            retrieved = sorted(retrieved, key=lambda x: x[0])

            from src.rerank.prompt import generative_multi_hop_filter_prompt

            messages = [
                {'role': 'system', 'content': generative_multi_hop_filter_prompt},
                {'role': 'user', 'content': f'\nQuery: {query}\nCandidate facts:\n' + '\n'.join([json.dumps(triple).lower() for triple in retrieved])},
                {'role': 'assistant', 'content': json.dumps({"fact": labels}).lower()}
            ]
            if split == 'train':
                training.append({"messages": messages})
            elif split == 'validation':
                validation.append({"messages": messages})
            else:
                test.append({"messages": messages})

    with open(training_output_path, 'w') as f:
        for item in training:
            f.write("%s\n" % json.dumps(item))
        print(f"Training data written to {training_output_path}")
    with open(validation_output_path, 'w') as f:
        for item in validation:
            f.write("%s\n" % json.dumps(item))
        print(f"Validation data written to {validation_output_path}")

    # calculate the number of tokens
    num_training_tokens = 0
    for item in training:
        num_training_tokens += num_tokens_by_tiktoken(json.dumps(item))
    print(f"Number of tokens in training data: {num_training_tokens}")
    num_validation_tokens = 0
    for item in validation:
        num_validation_tokens += num_tokens_by_tiktoken(json.dumps(item))
    print(f"Number of tokens in validation data: {num_validation_tokens}")

    if args.upload:
        client = OpenAI()
        client.fine_tuning.jobs.create(training_file=training_output_path, validation_file=validation_output_path, model=args.model, suffix=args.suffix)
        print("Fine-tuning job created, check https://platform.openai.com/finetune")
