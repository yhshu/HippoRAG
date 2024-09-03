import argparse
import json
import os
import random
from typing import List

from tqdm import tqdm


def format_question_and_candidates(question: str, candidates: List[str]):
    random.shuffle(candidates)
    question_and_candidates = f"Question: {question}\n\n"
    for i, p in enumerate(candidates):
        question_and_candidates += f"Passage {i + 1}: {p}\n\n"
    return question_and_candidates


def load_musique_data(data):
    """
    See examples: https://github.com/ContextualAI/gritlm/blob/main/gritlm/training/toy_data_instruct/toy_data_embedding.jsonl
    @param data:
    @return:
    """
    samples = []
    query_instruction = "Given a query and candidate passages, classify whether the passages fully support, partially support or do not support the query"
    doc_instruction = ""
    support_label = "The passages fully support the query."
    partial_support_label = "The passages partially support the query."
    non_support_label = "The passages do not support the query at all."
    for d in tqdm(data):
        question = d['question']
        supporting_passages = [item['title'] + '\n' + item['paragraph_text'] for item in d['paragraphs'] if item['is_supporting']]
        non_supporting_passages = [item['title'] + '\n' + item['paragraph_text'] for item in d['paragraphs'] if item['is_supporting'] is False]

        # 1. all supporting passages and some non-supporting passages, 5 in total
        candidate_passages = supporting_passages + random.sample(non_supporting_passages, 5 - len(supporting_passages))
        random.shuffle(candidate_passages)
        question_and_candidates = format_question_and_candidates(question, candidate_passages)
        samples.append({"query": [query_instruction, question_and_candidates], "pos": [[doc_instruction, support_label]],
                        "neg": [[doc_instruction, partial_support_label], [doc_instruction, non_support_label]]})

        # 2. partial supporting passages and some non-supporting passages, 5 in total
        num_elements = random.randint(1, min(3, len(supporting_passages) - 1))
        candidate_passages = random.sample(supporting_passages, num_elements) + random.sample(non_supporting_passages, 5 - num_elements)
        random.shuffle(candidate_passages)
        question_and_candidates = format_question_and_candidates(question, candidate_passages)
        samples.append({"query": [query_instruction, question_and_candidates], "pos": [[doc_instruction, partial_support_label]],
                        "neg": [[doc_instruction, non_support_label], [doc_instruction, support_label]]})

        # 3. all non-supporting passages, 5 in total
        candidate_passages = random.sample(non_supporting_passages, 5)
        question_and_candidates = format_question_and_candidates(question, candidate_passages)
        samples.append({"query": [query_instruction, question_and_candidates], "pos": [[doc_instruction, non_support_label]],
                        "neg": [[doc_instruction, support_label], [doc_instruction, partial_support_label]]})

    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GritLM/GritLM-7B')
    parser.add_argument('--datasets', nargs='+', type=str, help='A list of datasets')
    parser.add_argument('--exp', type=str, help='Experiment name', default='relevance_classifier')
    args = parser.parse_args()

    random.seed(1)
    with open('data/raw/musique/musique_ans_v1.0_train.jsonl', 'r') as f:
        train_data = f.readlines()
    train_data = [json.loads(d) for d in train_data]
    print('len train_data:', len(train_data))

    with open('data/raw/musique/musique_ans_v1.0_dev.jsonl', 'r') as f:
        dev_data = f.readlines()
    dev_data = [json.loads(d) for d in dev_data]
    print('len dev_data:', len(dev_data))

    train_samples = load_musique_data(train_data)
    dev_samples = load_musique_data(dev_data)

    os.makedirs('data/relevance_classifier_training/', exist_ok=True)
    with open('data/relevance_classifier_training/musique_train.jsonl', 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    with open('data/relevance_classifier_training/musique_dev.jsonl', 'w') as f:
        for sample in dev_samples:
            f.write(json.dumps(sample) + '\n')
