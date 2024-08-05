import sys

sys.path.append('.')

import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.openie.dataset_corpus_universal_ner import ner

entity_type_list = ['person', 'organization', 'location', 'product', 'event', 'law', 'country']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='Universal-NER/UniNER-7B-all')  # https://huggingface.co/Universal-NER
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    print(f"Using {torch.cuda.device_count()} devices")

    dataset = json.load(open(f'data/{args.dataset}.json'))
    queries_df = pd.read_json(f'data/{args.dataset}.json')

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model_label = args.model.split('/')[-1]

    ner_output_path = f'output/{args.dataset}_{model_label}_queries.named_entity_output.tsv'
    query_triples = []

    for sample in tqdm(dataset, desc='Extracting entities from queries', total=len(dataset)):
        query = sample['question']
        cur_ner_list = set()
        for t in entity_type_list:
            ner_results = ner(query, t, model, tokenizer, args.device)
            cur_ner_list.update(ner_results)
        query_triples.append({"named_entities": list(cur_ner_list)})

    queries_df['triples'] = query_triples
    queries_df.to_csv(ner_output_path, sep='\t')
    print('Query NER using UniversalNER saved to', ner_output_path)
