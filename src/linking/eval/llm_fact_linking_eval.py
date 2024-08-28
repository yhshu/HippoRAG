import sys

sys.path.append('.')

import argparse
import json

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from tqdm import tqdm

from src.langchain_util import init_langchain_model
from src.linking.llama3_fact_linker_train import load_custom_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='openai')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--datasets', nargs='+', type=str, help='A list of datasets, e.g., musique')
    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=f".llm_{args.model}_rerank.db"))
    if args.model.startswith('gpt-') or args.model.startswith('ft:gpt-'):
        model = init_langchain_model(args.llm, args.model)

    selected_datasets = args.datasets
    datasets = load_custom_dataset(selected_datasets=selected_datasets)
    metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    for idx, sample in tqdm(enumerate(datasets['validation']), total=len(datasets['validation']), desc='Evaluating'):
        messages = json.loads(sample['text'])
        try:
            gold_completion = eval(messages[2]['content']).get('fact', [])
            gold_completion = [tuple(t) for t in gold_completion]
        except Exception as e:
            print(e)
            continue

        try:
            completion = model.invoke(messages[:2], response_format={"type": "json_object"})
            content = json.loads(completion.content).get('fact', [])
            completion = [tuple(t) for t in content]
        except Exception as e:
            print(e)
            completion = []

        # Calculate precision, recall, and F1 between the gold completion and the generated completion
        precision = len(set(gold_completion) & set(completion)) / len(set(completion)) if len(set(completion)) > 0 else 0
        recall = len(set(gold_completion) & set(completion)) / len(set(gold_completion)) if len(set(gold_completion)) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics['precision'] += precision
        metrics['recall'] += recall
        metrics['f1'] += f1

        print('[GOLD]', gold_completion)
        print('[PREDICTION]', completion)
        for k, v in metrics.items():
            print(f'{k}: {round(v / (idx + 1), 4)}')
        print()

    for k, v in metrics.items():
        print(f'{k}: {round(v / len(datasets["validation"]), 4)}')
