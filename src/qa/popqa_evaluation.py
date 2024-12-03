import argparse
import json

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str)
    args = parser.parse_args()

    data = json.load(open(args.log))
    acc = 0
    for sample in tqdm(data):
        possible_answers = json.loads(sample['possible_answers'])
        prediction = sample['prediction']

        # if prediction is substring of any possible answer, increment accuracy
        if any(prediction.lower().strip('.') in possible_answer.lower() for possible_answer in possible_answers):
            acc += 1
        else:
            print(f'Question: {sample["question"]}')
            print(f'Prediction: {prediction}')
            print(f'Possible answers: {possible_answers}')
            print()

    print(f'Accuracy: {acc / len(data)}')