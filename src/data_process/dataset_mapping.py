import argparse
import json
import re

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import words, wordnet
from tqdm import tqdm

letter_mapping = {'A': 'E', 'B': 'P', 'C': 'K', 'D': 'T', 'E': 'I', 'F': 'V', 'G': 'C', 'H': 'R', 'I': 'O', 'J': 'L', 'K': 'G', 'L': 'Y', 'M': 'N', 'N': 'M', 'O': 'U', 'P': 'B',
                  'Q': 'Z', 'R': 'W', 'S': 'T', 'T': 'D', 'U': 'A', 'V': 'F', 'W': 'H', 'X': 'J', 'Y': 'L', 'Z': 'Q',
                  'a': 'e', 'b': 'p', 'c': 'k', 'd': 't', 'e': 'i', 'f': 'v', 'g': 'c', 'h': 'r', 'i': 'o', 'j': 'l', 'k': 'g', 'l': 'y', 'm': 'n', 'n': 'm', 'o': 'u', 'p': 'b',
                  'q': 'z', 'r': 'w', 's': 't', 't': 'd', 'u': 'a', 'v': 'f', 'w': 'h', 'x': 'j', 'y': 'l', 'z': 'q'}


def replace_text_with_mapping(text: str, mapping: dict):
    split_string = re.split('([^a-zA-Z]+)', text)
    for word in split_string:
        if word in mapping:
            text = text.replace(word, mapping[word])
    return text


def is_word_in_wordnet(word):
    return bool(wordnet.synsets(word))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    all_entities = json.load(open(f'data/universal_ner_{args.dataset}_Universal-NER_UniNER-7B-all_entities.json'))
    dataset = json.load(open(f'data/{args.dataset}.json'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))

    all_entities = set([e for t in all_entities for e in all_entities[t]])
    print(f"Loaded {len(all_entities)} entities")

    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    word_mapping = {}
    for e in tqdm(all_entities, total=len(all_entities), desc='Building word mapping'):
        for word in re.split('([^a-zA-Z]+)', e):
            if is_word_in_wordnet(word) or is_word_in_wordnet(word.lower()):
                continue
            if word.isupper():
                continue
            new_word = ''.join([letter_mapping[ch] if ch in letter_mapping else ch for ch in word])
            if new_word != word:
                word_mapping[word] = new_word

    print(f"Generated {len(word_mapping)} word mappings")

    num_modified_passage = 0
    for idx, passage in tqdm(enumerate(corpus), desc='Modifying corpus'):
        modified = False
        new_title = replace_text_with_mapping(passage['title'], word_mapping)
        if new_title != passage['title']:
            modified = True
            passage['title'] = new_title
        new_text = replace_text_with_mapping(passage['text'], word_mapping)
        if new_text != passage['text']:
            modified = True
            passage['text'] = new_text
        if 'idx' not in passage:
            passage['idx'] = str(idx)
        if modified:
            num_modified_passage += 1

    num_modified_question = 0
    for sample in tqdm(dataset, desc='Modifying dataset'):
        new_query = replace_text_with_mapping(sample['question'], word_mapping)
        if new_query != sample['question']:
            num_modified_question += 1
            sample['question'] = new_query
        for d in sample['question_decomposition']:
            d['answer'] = replace_text_with_mapping(d['answer'], word_mapping)
            d['question'] = replace_text_with_mapping(d['question'], word_mapping)
        for p in sample['paragraphs']:
            p['title'] = replace_text_with_mapping(p['title'], word_mapping)
            p['paragraph_text'] = replace_text_with_mapping(p['paragraph_text'], word_mapping)

    corpus_output_path = f'data/{args.dataset}_letter_mapping_corpus.json'
    json.dump(corpus, open(corpus_output_path, 'w'))
    print(f"Saved corpus to {corpus_output_path}, len: {len(corpus)}, num_modified: {num_modified_passage}")

    dataset_output = f'data/{args.dataset}_letter_mapping.json'
    json.dump(dataset, open(dataset_output, 'w'))
    print(f"Saved dataset to {dataset_output}, len: {len(dataset)}, num_modified: {num_modified_question}")
