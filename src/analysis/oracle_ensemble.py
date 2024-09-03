import argparse
import json
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log1', type=str)
    parser.add_argument('--log2', type=str)
    args = parser.parse_args()

    data1 = json.load(open(args.log1))
    data2 = json.load(open(args.log2))

    recall_k = 2
    ensemble = []
    win = []
    loss = []
    tie = []
    for i in range(len(data1)):
        sample1 = data1[i]
        sample2 = data2[i]
        question_key = 'question' if 'question' in sample1 else 'query'
        assert sample1[question_key] == sample2[question_key]

        if 'recall' in sample1 and 'recall' in sample2:
            metric1 = round(sample1['recall'][str(recall_k)], 3)
            metric2 = round(sample2['recall'][str(recall_k)], 3)
        elif 'ndcg' in sample1 and 'ndcg' in sample2:
            metric1 = round(sample1['ndcg'], 3)
            metric2 = round(sample2['ndcg'], 3)

        if metric1 > metric2:
            ensemble.append(sample1)
            win.append(sample1)
        elif metric1 < metric2:
            ensemble.append(sample2)
            loss.append(sample2)
        else:
            ensemble.append(sample1)
            tie.append(sample1)

    # print sample info
    print('WIN')
    for sample in win[:20]:
        print(f'win: {sample[question_key]}')
    print('\nLOSS')
    for sample in loss[:20]:
        print(f'loss: {sample[question_key]}')
    print('\nTIE')
    for sample in tie[:20]:
        print(f'tie: {sample[question_key]}')

    # evaluate ensemble performance
    metrics = defaultdict(float)
    for sample in ensemble:
        if 'recall' in sample:
            for key, value in sample['recall'].items():
                metrics[key] += value
        if 'ndcg' in sample:
            metrics['ndcg'] += sample['ndcg']

    for key in metrics:
        metrics[key] /= len(ensemble)
        print(f'{key}: {round(metrics[key], 3)}')
    print(f'num_select1: {len(win) / len(ensemble)}')
    print(f'num_select2: {len(loss) / len(ensemble)}')
    print(f'num_tie: {len(tie) / len(ensemble)}')
    print(f'total: {len(ensemble)}')
