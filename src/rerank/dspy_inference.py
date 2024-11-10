import argparse
import json

import dspy

from src.rerank.dspy_optimize import  Filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-4o', help='Language model to use')
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str)
    args = parser.parse_args()

    program = Filter()
    url = f'http://{args.addr}:{args.port}/v1'
    dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=3000, temperature=0.0, api_base=url, api_key='osunlp')
    dspy.settings.configure(lm=dspy_llm)
    program.load("output/dspy/fact_filter_mipro_optimized_meta-llama_Llama-3.1-70B-Instruct_predict_400_79.json")

    fact_before_filter = {"fact": [["Paris", "is the capital of", "France", "Berlin", "is the capital of", "Germany"]]}
    print(program(question="What is the capital of France?", fact_before_filter=json.dumps(fact_before_filter)))
