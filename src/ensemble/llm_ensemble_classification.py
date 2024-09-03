import argparse
import json

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import SystemMessage, HumanMessage

from src.langchain_util import init_langchain_model

system_message = """Given a query and candidate passages, classify whether the passages fully support, partially support or do not support the query. Consider multi-hop reasoning across passages. Respond in JSON format, e.g., {"label": "FULLY"}, {"label": "PARTIALLY"}, or {"label": "NOT"}."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

    client = init_langchain_model(args.llm, args.model)

    with open('data/relevance_classifier_training/musique_dev.jsonl', 'r') as f:
        dev_data = f.readlines()
    dev_data = [json.loads(d) for d in dev_data]

    set_llm_cache(SQLiteCache(database_path=f".ensemble_classify_{args.model}.db"))

    acc = 0
    for sample in dev_data[:100]:
        messages = [SystemMessage(system_message), HumanMessage(sample['query'][1])]
        gold_label = sample['pos'][0][1]
        if 'fully' in gold_label:
            gold_label = 'FULLY'
        elif 'partially' in gold_label:
            gold_label = 'PARTIALLY'
        else:
            gold_label = 'NOT'

        pred_label = None
        completions = client.invoke(messages, temperature=0, response_format={"type": "json_object"})
        content = completions.content
        if content:
            pred_label = json.loads(content)['label']
        if pred_label == gold_label:
            acc += 1

        print(f"Gold: {gold_label}, Pred: {pred_label}")

    print(f"Accuracy: {acc / 100}")
