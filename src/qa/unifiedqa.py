import argparse
import json

import evaluate
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def run_model(input_string, device='cuda', **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids.to(device)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def normalize_text(text):
    import tensorflow as tf

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
    parser.add_argument('-nr', '--no_retrieval', action='store_true', help='whether to use retrieval')
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
        if args.no_retrieval:
            retrieved = []
        prediction = run_model(format_input(question, retrieved), device=device)
        predictions.append(prediction[0])
        references.append(gold_ans)

    bleu1_results = bleu.compute(predictions=predictions, references=references, max_order=1)
    bleu4_results = bleu.compute(predictions=predictions, references=references, max_order=4)
    rouge_results = rouge.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)

    # for each dict, print float with 4 decimal places
    for metric_dict in [bleu1_results, bleu4_results, rouge_results, meteor_results]:
        for key, value in metric_dict.items():
            if isinstance(value, float):
                metric_dict[key] = round(value, 4)
            elif isinstance(value, list):
                metric_dict[key] = [round(v, 4) for v in value]
        print(metric_dict)
