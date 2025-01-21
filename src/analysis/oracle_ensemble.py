import argparse
import json
import os
from collections import defaultdict


def statistics(samples, metrics, split: str):
    for sample in samples:
        if 'question_decomposition' in sample:
            metrics['num_hop'] += len(sample['question_decomposition'])
        if 'type' in sample:
            metrics[sample['type']] += 1
        metrics['metric1'] += sample['metric1']
        metrics['metric2'] += sample['metric2']
        metrics['question_str_len'] += len(sample[question_key])
    for key in metrics:
        metrics[key] /= len(samples)
        print(f'{split}_{key}: {round(metrics[key], 3)}')
    print(f'len {split}: {len(samples)}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log1', type=str)
    parser.add_argument('--log2', type=str)
    args = parser.parse_args()

    data1 = json.load(open(args.log1))
    data2 = json.load(open(args.log2))

    assert len(data1) == len(data2), f'Two logs have different number of samples: {len(data1)} vs {len(data2)}'

    recall_k = 5
    ensemble = []
    win = []
    loss = []
    tie = []
    for i in range(len(data1)):
        sample1 = data1[i]
        sample2 = data2[i]
        question_key = 'question' if 'question' in sample1 else 'query'
        assert sample1[question_key] == sample2[question_key], f'Two queries at {i} are not the same: {sample1[question_key]}\t{sample2[question_key]}'

        if 'recall' in sample1 and 'recall' in sample2:
            metric1 = round(sample1['recall'][str(recall_k)], 3)
            metric2 = round(sample2['recall'][str(recall_k)], 3)
        elif 'ndcg' in sample1 and 'ndcg' in sample2:
            metric1 = round(sample1['ndcg'], 3)
            metric2 = round(sample2['ndcg'], 3)

        if metric1 > metric2:
            sample1['metric1'] = metric1
            sample1['metric2'] = metric2
            sample1['retrieved1'] = sample1['retrieved']
            sample1['retrieved2'] = sample2['retrieved']
            del sample1['retrieved']
            ensemble.append(sample1)
            win.append(sample1)
        elif metric1 < metric2:
            sample2['metric1'] = metric1
            sample2['metric2'] = metric2
            sample2['retrieved1'] = sample1['retrieved']
            sample2['retrieved2'] = sample2['retrieved']
            del sample2['retrieved']
            ensemble.append(sample2)
            loss.append(sample2)
        else:
            sample1['metric1'] = metric1
            sample1['metric2'] = metric2
            sample1['retrieved1'] = sample1['retrieved']
            sample2['retrieved2'] = sample2['retrieved']
            del sample1['retrieved']
            ensemble.append(sample1)
            tie.append(sample1)

    # print sample info
    print('WIN')
    for sample in win:
        type_str = f'[{sample["type"]}]' if 'type' in sample else ''
        print(f"win [{sample['metric1']} > {sample['metric2']}] {type_str} {sample[question_key]}")
    print('\nLOSS')
    for sample in loss:
        type_str = f'[{sample["type"]}]' if 'type' in sample else ''
        print(f"loss [{sample['metric1']} < {sample['metric2']}] {type_str} {sample[question_key]}")
    print('\nTIE')
    for sample in tie:
        type_str = f'[{sample["type"]}]' if 'type' in sample else ''
        print(f"tie [{sample['metric1']} = {sample['metric2']}] {type_str} {sample[question_key]}")

    # evaluate ensemble performance
    metrics = defaultdict(float)
    for sample in ensemble:
        if 'recall' in sample:
            for key, value in sample['recall'].items():
                metrics[key] += value
        if 'ndcg' in sample:
            metrics['ndcg'] += sample['ndcg']
        if 'type' in sample:
            metrics[sample['type']] += 1

    for key in metrics:
        metrics[key] /= len(ensemble)
        print(f'{key}: {round(metrics[key], 3)}')
    print(f'select 1: {len(win) * 100 / len(ensemble)}%, select 2: {len(loss) * 100 / len(ensemble)}%, tie: {len(tie) * 100 / len(ensemble)}%')
    print(f'total: {len(ensemble)}')
    print()

    # calculate the average metric1 and metric2 for win, loss, and tie
    metrics_win = defaultdict(float)
    metrics_loss = defaultdict(float)
    metrics_tie = defaultdict(float)

    statistics(win, metrics_win, 'win')
    statistics(loss, metrics_loss, 'loss')
    statistics(tie, metrics_tie, 'tie')

    os.makedirs('output/compare', exist_ok=True)
    with open('output/compare/win.json', 'w') as f:
        json.dump(win, f, indent=2)
    with open('output/compare/loss.json', 'w') as f:
        json.dump(loss, f, indent=2)
    with open('output/compare/tie.json', 'w') as f:
        json.dump(tie, f, indent=2)
