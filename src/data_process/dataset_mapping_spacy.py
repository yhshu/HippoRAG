import argparse
import json

import spacy
from tqdm import tqdm

from src.data_process.dataset_mapping_universal_ner import replace_dataset_and_corpus


def spacy_extract_entities(text):
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'EVENT', 'LAW', 'LOC']:
            entities.add(ent.text)
    return entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    dataset = json.load(open(f'data/{args.dataset}.json'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))

    all_entities = set()

    for sample in tqdm(dataset):
        question = sample['question']
        all_entities.update(spacy_extract_entities(question))

        for d in sample['question_decomposition']:
            all_entities.update(spacy_extract_entities(d['question']))
            all_entities.update(spacy_extract_entities(d['answer']))

    for passage in tqdm(corpus):
        all_entities.update(spacy_extract_entities(passage['title']))
        all_entities.update(spacy_extract_entities(passage['text']))

    print(f"{len(all_entities)} entities extracted")

    replace_dataset_and_corpus(corpus, dataset, all_entities, args.dataset)
