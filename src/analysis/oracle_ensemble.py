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

    recall_k = 5
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
            sample1['metric1'] = metric1
            sample1['metric2'] = metric2
            ensemble.append(sample1)
            win.append(sample1)
        elif metric1 < metric2:
            sample2['metric1'] = metric1
            sample2['metric2'] = metric2
            ensemble.append(sample2)
            loss.append(sample2)
        else:
            sample1['metric1'] = metric1
            sample1['metric2'] = metric2
            ensemble.append(sample1)
            tie.append(sample1)

    # print sample info
    print('WIN')
    for sample in win:
        print(f"win [{sample['metric1']} > {sample['metric2']}] {sample[question_key]}")
    print('\nLOSS')
    for sample in loss:
        print(f"loss [{sample['metric1']} < {sample['metric2']}] {sample[question_key]}")
    print('\nTIE')
    for sample in tie:
        print(f"tie [{sample['metric1']} = {sample['metric2']}] {sample[question_key]}")

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

    # calculate the average metric1 and metric2 for win, loss, and tie
    metrics_win = defaultdict(float)
    for sample in win:
        if 'question_decomposition' in sample:
            metrics_win['num_hop'] += len(sample['question_decomposition'])
        metrics_win['metric1'] += sample['metric1']
        metrics_win['metric2'] += sample['metric2']
        metrics_win['question_str_len'] += len(sample[question_key])
    for key in metrics_win:
        metrics_win[key] /= len(win)
        print(f'win_{key}: {round(metrics_win[key], 3)}')
    print('len win:', len(win))
    print()

    metrics_loss = defaultdict(float)
    for sample in loss:
        if 'question_decomposition' in sample:
            metrics_loss['num_hop'] += len(sample['question_decomposition'])
        metrics_loss['metric1'] += sample['metric1']
        metrics_loss['metric2'] += sample['metric2']
        metrics_loss['question_str_len'] += len(sample[question_key])
    for key in metrics_loss:
        metrics_loss[key] /= len(loss)
        print(f'loss_{key}: {round(metrics_loss[key], 3)}')
    print('len loss:', len(loss))
    print()

    metrics_tie = defaultdict(float)
    for sample in tie:
        if 'question_decomposition' in sample:
            metrics_tie['num_hop'] += len(sample['question_decomposition'])
        metrics_tie['metric1'] += sample['metric1']
        metrics_tie['question_str_len'] += len(sample[question_key])
    for key in metrics_tie:
        metrics_tie[key] /= len(tie)
        print(f'tie_{key}: {round(metrics_tie[key], 3)}')
    print('len tie:', len(tie))