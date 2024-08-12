import argparse
import json


def compare_recalls(data1, data2):
    win = []
    loss = []
    tie = []

    win_recall1 = []
    win_recall2 = []
    loss_recall1 = []
    loss_recall2 = []
    for record1, record2 in zip(data1, data2):
        if record1['recall']['5'] > record2['recall']['5']:
            win.append(record1)
            win_recall1.append(record1['recall']['5'])
            win_recall2.append(record2['recall']['5'])
        elif record1['recall']['5'] < record2['recall']['5']:
            loss.append(record2)
            loss_recall1.append(record1['recall']['5'])
            loss_recall2.append(record2['recall']['5'])
        else:
            tie.append(record1)

    win_len_question = sum([len(record['question']) for record in win]) / len(win)
    win_num_step = sum([len(record['question_decomposition']) for record in win]) / len(win)

    loss_len_question = sum([len(record['question']) for record in loss]) / len(loss)
    loss_num_step = sum([len(record['question_decomposition']) for record in loss]) / len(loss)

    tie_len_question = sum([len(record['question']) for record in tie]) / len(tie)
    tie_num_step = sum([len(record['question_decomposition']) for record in tie]) / len(tie)

    tie_avg_recall = sum([record['recall']['5'] for record in tie]) / len(tie)
    print('win:', len(win))
    print('win len question:', win_len_question)
    print('win num step:', win_num_step)
    print('win avg recall 1:', sum(win_recall1) / len(win))
    print('win avg recall 2:', sum(win_recall2) / len(win))
    print('loss:', len(loss))
    print('loss len question:', loss_len_question)
    print('loss num step:', loss_num_step)
    print('loss avg recall 1:', sum(loss_recall1) / len(loss))
    print('loss avg recall 2:', sum(loss_recall2) / len(loss))
    print('tie:', len(tie))
    print('tie len question:', tie_len_question)
    print('tie num step:', tie_num_step)
    print('tie avg recall:', tie_avg_recall)

    win_questions = [record['question'] for record in win]
    loss_questions = [record['question'] for record in loss]
    tie_questions = [record['question'] for record in tie]
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

    win, loss, tie = compare_recalls(data1, data2)
    label1 = args.data1.split('/')[-1].split('.')[0]
    label2 = args.data2.split('/')[-1].split('.')[0]
    json.dump(win, open(f'output/win_{label1}_vs_{label2}.json', 'w'), indent=4)
    json.dump(loss, open(f'output/loss_{label1}_vs_{label2}.json', 'w'), indent=4)
    json.dump(tie, open(f'output/tie_{label1}_vs_{label2}.json', 'w'), indent=4)
