import argparse
import json

import evaluate
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, help='the file path to QA results')
    args = parser.parse_args()

    data = json.load(open(args.log))

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    predictions = []
    references = []
    for sample in tqdm(data):
        question = sample['question']
        gold_ans = sample['answer']

        retrieved = sample['retrieved']
        prediction = ''
        if isinstance(prediction, list):
            prediction = prediction[0]
        elif isinstance(prediction, str):
            prediction = sample['prediction']
        else:
            print('Invalid prediction type:', type(prediction))
        predictions.append(prediction)
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
