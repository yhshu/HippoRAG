import argparse
import json
import os

import dspy
from dspy import Evaluate
from dspy.teleprompt import MIPROv2
from pydantic import BaseModel, Field
from sympy.physics.units import temperature


class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class FactFiltering(dspy.Signature):
    question = dspy.InputField(desc="Query for retrieval")
    fact_before_filter = dspy.InputField(desc="Candidate facts to be filtered")
    fact_after_filter: Fact = dspy.OutputField(desc="Filtered facts in JSON format")


class Filter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.TypedChainOfThought(FactFiltering)

    def forward(self, question, fact_before_filter):
        return self.prog(question=question, fact_before_filter=fact_before_filter)


def filtering_precision(example, pred, trace=None):
    if len(pred) == 0:
        return 0
    gold = example.fact_after_filter

    try:
        pred_list = pred.fact_after_filter.fact
        gold_list = json.loads(gold).get('fact', [])
    except Exception as e:
        print(f"Error: {e}")
        pred_list = []
        gold_list = []

    if len(gold_list) == 0:
        return 0

    gold_set = set([tuple(t) for t in gold_list])
    pred_set = set([tuple(t) for t in pred_list])
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1
    elif len(pred_set) == 0:
        return 0
    return len(gold_set.intersection(pred_set)) / len(pred_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-4o', help='Language model to use')
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str)
    args = parser.parse_args()

    if args.llm.startswith('gpt-'):
        dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=3000, temperature=0.0)
    elif args.addr is not None and args.port is not None:
        url = f'http://{args.addr}:{args.port}/v1'
        dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=3000, api_base=url, api_key='osunlp')
    else:
        raise ValueError(f"LM not implemented: {args.llm}")
    dspy.settings.configure(lm=dspy_llm)

    train = json.load(open('data/fact_filter/train.json'))
    dev = json.load(open('data/fact_filter/dev.json'))

    from dspy.datasets import DataLoader

    dl = DataLoader()
    trainset = dl.from_json('data/fact_filter/train.json',
                            fields=("question", "fact_before_filter", "fact_after_filter"),
                            input_keys=("question", "fact_before_filter"))
    devset = dl.from_json('data/fact_filter/dev.json',
                          fields=("question", "fact_before_filter", "fact_after_filter"),
                          input_keys=("question", "fact_before_filter"))

    filter_metric = filtering_precision
    evaluate = Evaluate(devset=devset[:], metric=filter_metric, num_threads=8, display_progress=True, display_table=False)

    # Initialize optimizer
    teleprompter = MIPROv2(
        metric=filter_metric,
        auto="light",  # Can choose between light, medium, and heavy optimization runs
    )
    program = Filter()

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        requires_permission_to_run=False,
    )

    # Save optimize program for future use
    os.makedirs("output/dspy", exist_ok=True)
    model_label = args.llm.replace("/", "_")
    optimized_program.save(f"output/dspy/fact_filter_mipro_optimized_{model_label}.json")

    # Evaluate optimized program
    print(f"Evaluate optimized program...")
    evaluate(optimized_program, devset=devset[:])
