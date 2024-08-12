import argparse
import json
import os


def compare_ndcg(data1, data2):
    win = []
    loss = []
    tie = []

    win_ndcg1 = []
    win_ndcg2 = []
    loss_ndcg1 = []
    loss_ndcg2 = []
    tie_ndcg = []

    for record1, record2 in zip(data1, data2):
        assert record1['query'] == record2['query']
        ndcg_1 = record1['ndcg']
        ndcg_2 = record2['ndcg']
        if ndcg_1 > ndcg_2:
            win.append(record1)
            win_ndcg1.append(ndcg_1)
            win_ndcg2.append(ndcg_2)
        elif ndcg_1 < ndcg_2:
            loss.append(record2)
            loss_ndcg1.append(ndcg_1)
            loss_ndcg2.append(ndcg_2)
        else:
            tie.append(record1)
            tie_ndcg.append(ndcg_1)

    win_len_question = sum([len(record['query']) for record in win]) / len(win)
    loss_len_question = sum([len(record['query']) for record in loss]) / len(loss)
    tie_len_question = sum([len(record['query']) for record in tie]) / len(tie)

    print('win:', len(win))
    print('win len question:', win_len_question)
    print('win avg ndcg 1:', sum(win_ndcg1) / len(win))
    print('win avg ndcg 2:', sum(win_ndcg2) / len(win))
    print('loss:', len(loss))
    print('loss len question:', loss_len_question)
    print('loss avg ndcg 1:', sum(loss_ndcg1) / len(loss))
    print('loss avg ndcg 2:', sum(loss_ndcg2) / len(loss))
    print('tie:', len(tie))
    print('tie len question:', tie_len_question)
    print('tie avg ndcg:', sum(tie_ndcg) / len(tie))

    win_questions = [record['query'] for record in win]
    loss_questions = [record['query'] for record in loss]
    tie_questions = [record['query'] for record in tie]
    return win_questions, loss_questions, tie_questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1", type=str, required=True, help="Path to the first JSON file.")
    parser.add_argument("--data2", type=str, required=True, help="Path to the second JSON file.")
    args = parser.parse_args()

    with open(args.data1) as f:
        data1 = json.load(f)
        if isinstance(data1, dict):
            data1 = data1['log']

    with open(args.data2) as f:
        data2 = json.load(f)

    win, loss, tie = compare_ndcg(data1, data2)
    label1 = args.data1.split('/')[-1].split('.')[0]
    label2 = args.data2.split('/')[-1].split('.')[0]

    os.makedirs('output/win_loss', exist_ok=True)
    json.dump(win, open(f'output/win_loss/win_{label1}_vs_{label2}.json', 'w'), indent=4)
    json.dump(loss, open(f'output/win_loss/loss_{label1}_vs_{label2}.json', 'w'), indent=4)
    json.dump(tie, open(f'output/win_loss/tie_{label1}_vs_{label2}.json', 'w'), indent=4)
