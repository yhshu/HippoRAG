import argparse
import json
from collections import defaultdict

import evaluate
import tensorflow as tf
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def run_model(input_string, device='cuda', **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids.to(device)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
    return text


def format_input(question: str, passages: list):
    passage_str = '\n'.join(passages)
    input_string = f"{question} \\n {passage_str}"
    return input_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unifiedqa_model', default='allenai/unifiedqa-v2-t5-3b-1251000', type=str)
    parser.add_argument('--context', type=str, help='the file path to retrieval context')
    args = parser.parse_args()

    device = 'cuda'
    tokenizer = T5Tokenizer.from_pretrained(args.unifiedqa_model)
    model = T5ForConditionalGeneration.from_pretrained(args.unifiedqa_model).to(device)

    data = json.load(open(args.context))

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    predictions = []
    references = []
    for sample in tqdm(data):
        question = sample['question']
        gold_ans = sample['answer']

        retrieved = sample['retrieved']
        prediction = run_model(format_input(question, retrieved), device=device)
        predictions.append(prediction[0])
        references.append(gold_ans)

    bleu_results = bleu.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)

    print(bleu_results)
    print(rouge_results)
    print(meteor_results)
