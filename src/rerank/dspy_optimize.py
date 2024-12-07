import argparse
import json
import os

import dspy
from dspy import Evaluate
from dspy.teleprompt import MIPROv2
from pydantic import BaseModel, Field


class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class FactWithConfidence(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")
    confidence: str = Field(description="Confidence level of the generated facts, either 'high' or 'low'")


class FactFilteringSignature(dspy.Signature):
    """
   Filter facts based on their relevance to the query. Carefully generate related facts from the candidate list that have strong connection to the query.

   - Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
   - The relevance means the fact aids in reasoning and providing an answer.
   - Select up to 4 relevant facts from the candidate list and output in JSON format without any other words, e.g.,

   ```json
   {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
   ```

   - If no facts are relevant, return an empty list, e.g., {"fact": []}.
   - Only use facts from the candidate list; do NOT generate new facts.
   """
    question = dspy.InputField(desc="Query for retrieval")
    fact_before_filter = dspy.InputField(desc="Candidate facts to be filtered")
    fact_after_filter: Fact = dspy.OutputField(desc="Filtered facts in JSON format")


class FactFilteringWithConfidenceSignature(dspy.Signature):
    """
    Filter facts based on their relevance to the query. Carefully generate related facts from the candidate list that have strong connection to the query.

    - Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
    - If the query is a claim, relevance means the fact supports or contradicts it.
    - For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
    - Select up to 4 relevant facts from the candidate list, and tell the confidence of your generated facts using "high" or "low" categories.

    - Output in JSON format without any other words, e.g.,

    ```json
    {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]], "confidence": "low"}.
    ```

    - If no facts are relevant, return an empty list, e.g., {"fact": [], "confidence": "high"}.
    - Only use facts from the candidate list; do NOT generate new facts.
    """
    question = dspy.InputField(desc="Query for retrieval")
    fact_before_filter = dspy.InputField(desc="Candidate facts to be filtered")
    fact_after_filter: FactWithConfidence = dspy.OutputField(desc="Filtered facts in JSON format")


class FactFilterProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(FactFilteringSignature)
        # self.prog = dspy.Predict(FactFilteringWithConfidenceSignature)

    def forward(self, question, fact_before_filter):
        try:
            return self.prog(question=question, fact_before_filter=fact_before_filter)
        except Exception as e:
            print(f"Filter forward exception: {e}")
            from dspy.primitives.prediction import Prediction
            return Prediction(fact_after_filter=Fact(fact=[]))


def filtering_precision(example, pred, trace=None):
    try:
        if len(pred) == 0:
            pred_list = []
        else:
            pred_list = pred.fact_after_filter.fact
    except Exception as e:
        print(f"Error: {e}")
        pred_list = []
    try:
        gold = example.fact_after_filter
        gold_list = json.loads(gold).get('fact', [])
    except Exception as e:
        print(f"Error: {e}")
        gold_list = []

    gold_set = set([tuple(t) for t in gold_list])
    pred_set = set([tuple(t) for t in pred_list])
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1
    elif len(pred_set) == 0 and len(gold_set) > 0:
        return 0
    elif len(gold_set) == 0 and len(pred_set) > 0:
        return 0

    if trace is None:
        return len(gold_set.intersection(pred_set)) / len(pred_set)
    else:
        return gold_set == pred_set


def filtering_recall(example, pred, trace=None):
    try:
        if len(pred) == 0:
            pred_list = []
        else:
            pred_list = pred.fact_after_filter.fact
    except Exception as e:
        print(f"Error: {e}")
        pred_list = []
    try:
        gold = example.fact_after_filter
        gold_list = json.loads(gold).get('fact', [])
    except Exception as e:
        print(f"Error: {e}")
        gold_list = []

    gold_set = set([tuple(t) for t in gold_list])
    pred_set = set([tuple(t) for t in pred_list])
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1
    elif len(pred_set) == 0 and len(gold_set) > 0:
        return 0
    elif len(gold_set) == 0 and len(pred_set) > 0:
        return 1
    return len(gold_set.intersection(pred_set)) / len(gold_set)


def filtering_f1(example, pred, trace=None):
    p = filtering_precision(example, pred, trace)
    r = filtering_recall(example, pred, trace)
    if p + r == 0:
        return 0
    f1 = 2 * p * r / (p + r)
    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-4o', help='Language model to use')
    parser.add_argument('--addr', type=str, default='localhost')
    parser.add_argument('--port', type=str)
    parser.add_argument('--auto', type=str, default='light', help='Optimization level')
    args = parser.parse_args()

    if args.llm.startswith('gpt-'):
        dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=256, temperature=0.0)
    elif args.addr is not None and args.port is not None:
        url = f'http://{args.addr}:{args.port}/v1'
        dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=256, temperature=0.0, api_base=url, api_key='osunlp')
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
        auto=args.auto,  # Can choose between light, medium, and heavy optimization runs
    )
    program = FactFilterProgram()

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        valset=devset,
        max_bootstrapped_demos=10,
        max_labeled_demos=10,
        requires_permission_to_run=False,
    )

    # Save optimize program for future use
    os.makedirs("output/dspy", exist_ok=True)
    model_label = args.llm.replace("/", "_")
    output_path = f"output/dspy/fact_filter_mipro_optimized_{model_label}.json"
    optimized_program.save(output_path)

    # Evaluate optimized program
    print(f"Evaluate optimized program...")
    evaluate(optimized_program, devset=devset[:])
    print(f"Optimized program saved at {output_path}")
