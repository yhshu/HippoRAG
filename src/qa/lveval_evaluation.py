import argparse
import json
import re
import string
from collections import Counter

ABANDON_WORDS_EN = ['and', 'to', 'of', 'in', 'her', 'was', 'with', 'for', 'it', 'from', 'is', 'that', 'his', 'he', 'by', 'she', 'they', 'or', 'at', 'because', 'be', 'on', 'are',
                    'their', 'what', 'as', 'had', 'were', 'about', 'being', 'this', 'who', 'but', 'have', 'has', 'when', 'which', 'does']


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score_with_gold_ans(prediction, ground_truth, gold_ans=None, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    # answer keywords recall
    if gold_ans:
        gold_ans_tokens = normalize_answer(gold_ans)
        gold_ans_tokens = gold_ans_tokens.split()
        common = Counter(prediction_tokens) & Counter(gold_ans_tokens)
        filtered_common = {key: value for key, value in common.items() if key not in ABANDON_WORDS_EN}
        num_same = sum(filtered_common.values())
        recall = 1.0 * num_same / len(gold_ans_tokens)
        if recall < 0.2: return 0.

    return f1_score(prediction_tokens, ground_truth_tokens)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None)
    return parser.parse_args(args)


def custom_sort(s):
    letters = re.findall('[a-zA-Z]+', s)
    numbers = re.findall('\d+', s)
    return (letters, int(numbers[0])) if numbers else (letters, 0)


def scorer(predictions, answers, gold_anss, metric=qa_f1_score_with_gold_ans):
    total_score = 0.
    total_sample = 0
    scores = {metric.__name__: []}
    for (prediction, ground_truths, gold_ans) in zip(predictions, answers, gold_anss):
        total_sample += 1
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, metric(prediction, ground_truth, gold_ans))
            break
        total_score += score
        scores[metric.__name__].append(score)
    return round(100 * total_score / total_sample, 2), scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str)
    args = parser.parse_args()

    log = json.load(open(args.log))
    predictions = []
    answers = []
    gold_anss = []
    for sample in log:
        predictions.append(sample['prediction'])
        answers.append(sample['answers'])
        gold_anss.append(sample['gold_ans'] if 'gold_ans' in sample else None)

    res = scorer(predictions, answers, gold_anss)
    print(res)
