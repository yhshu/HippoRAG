import argparse
import json
import re

import pandas as pd
import spacy
from tqdm import tqdm

import random
import string


def random_char_replacement(name, num_replacements=2):
    name_list = list(name)
    for _ in range(num_replacements):
        index = random.randint(1, len(name_list) - 1)
        if name_list[index] in string.ascii_lowercase:
            name_list[index] = random.choice(string.ascii_lowercase)
        elif name_list[index] in string.ascii_uppercase:
            name_list[index] = random.choice(string.ascii_uppercase)
    return ''.join(name_list)


def add_prefix_suffix(name, prefix_length=0, suffix_length=2):
    prefix = ''.join(random.choices(string.ascii_letters, k=prefix_length))
    suffix = ''.join(random.choices(string.ascii_lowercase, k=suffix_length))
    return f"{prefix}{name}{suffix}"


def random_insert_char(name, num_inserts=2):
    name_list = list(name)
    for _ in range(num_inserts):
        index = random.randint(1, len(name_list))
        name_list.insert(index, random.choice(string.ascii_lowercase))
    return ''.join(name_list)


def mix_string(name, keep_ratio=0.5):
    name_list = list(name)
    num_keep = int(len(name_list) * keep_ratio)
    keep_indices = random.sample(range(1, len(name_list)), num_keep - 1) + [0]
    mixed_name = [name_list[i] if i in keep_indices else random.choice(string.ascii_lowercase) for i in range(len(name_list))]
    return ''.join(mixed_name)


def perturb_entity(name):
    length = len(name)
    num_words = len(name.split())

    if length <= 4 or num_words == 1:
        return add_prefix_suffix(name)
    elif 5 <= length <= 10:
        method = random.choice([random_char_replacement, random_insert_char])
        return method(name)
    else:
        method = random.choice([mix_string, random_insert_char])
        return method(name)


def perturb_entity_set(entities):
    # remove numbers from entities
    entities = [e for e in entities if not is_number(e)]

    short_entities = [e for e in entities if len(e) <= 4 or len(e.split()) == 1]
    long_entities = [e for e in entities if e not in short_entities]

    mapping = {}

    # process short entities
    for entity in short_entities:
        perturbed = perturb_entity(entity)
        mapping[entity] = perturbed

    # process long entities
    for entity in long_entities:
        perturbed_entity = entity
        for short_entity in short_entities:
            if short_entity in entity:
                perturbed_entity = perturbed_entity.replace(short_entity, mapping[short_entity])

        # determine the remaining part
        remaining_part = ' '.join([word for word in perturbed_entity.split() if word not in mapping.values()])
        if remaining_part:
            perturbed_remaining = perturb_entity(remaining_part)
            perturbed_entity = perturbed_entity.replace(remaining_part, perturbed_remaining)

        mapping[entity] = perturbed_entity

    return mapping


def is_number(s):
    s = s.strip()
    if s.isdigit():
        return True
    roman_numerals = {'I', 'V', 'X', 'L', 'C', 'D', 'M'}
    if all(c in roman_numerals for c in s):
        return True

    if re.match(r'^\d+([.,]\d+)*$', s):
        return True

    return False


def replace_phrases(text, mapping):
    # Sort the mapping keys by length in descending order to handle longer phrases first
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    def replace_match(match):
        matched_text = match.group(0)
        return mapping.get(matched_text, matched_text)

    # Create a regex pattern that matches any of the keys in the mapping
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in sorted_keys) + r')\b')

    # Replace matched phrases using the replace_match function
    replaced_text = pattern.sub(replace_match, text)

    return replaced_text


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    args = paser.parse_args()

    random.seed(1)

    musique_samples = json.load(open('data/musique.json'))
    musique_corpus = json.load(open('data/musique_corpus.json'))
    musique_query_ner = pd.read_csv('output/musique_queries.named_entity_output.tsv', sep='\t')

    # all_named_entities = set()
    nlp = spacy.load("en_core_web_sm")
    entity_types = dict()
    for index, row in tqdm(musique_query_ner.iterrows()):
        query = row['question']
        named_entities = json.loads(row['triples'])['named_entities']
        # all_named_entities.update(named_entities)
        doc = nlp(query)
        if doc.ents is None or len(doc.ents) == 0:
            print(f"No entitiy found in query by Spacy: {query}")
            print(f"NER by GPT-3.5: {named_entities}")
            print()
        for ent in doc.ents:
            if ent.label_ not in entity_types:
                entity_types[ent.label_] = set()
            entity_types[ent.label_].add(ent.text)

    intermediate_entities = dict()
    for sample in musique_samples:
        decomposition = sample['question_decomposition']
        for d in decomposition:
            doc = nlp(d['answer'])
            if doc.ents is None or len(doc.ents) == 0:
                print(f"No entitiy found in query by Spacy: {d['answer']}")
            for ent in doc.ents:
                if ent.label_ not in intermediate_entities:
                    intermediate_entities[ent.label_] = set()
                intermediate_entities[ent.label_].add(ent.text)

    # statistics
    all_entities = set()
    count = 0
    for key in entity_types:
        print(key, len(entity_types[key]))
        if key == 'PERSON':
            for e in entity_types[key]:
                if len(e.split(' ')) > 1:
                    count += 1
        if key in ['PERSON', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'EVENT', 'LAW']:
            all_entities.update(entity_types[key])
    print('Person name more than one word from query entities', count)

    count = 0
    for key in intermediate_entities:
        print(key, len(intermediate_entities[key]))
        if key == 'PERSON':
            for e in intermediate_entities[key]:
                if len(e.split(' ')) > 1:
                    count += 1
        if key in ['PERSON', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'EVENT', 'LAW']:
            all_entities.update(intermediate_entities[key])
    print('Person name more than one word from intermediate answers', count)

    # perturb the entities
    print('Collected entities:', len(all_entities))
    entity_mapping = perturb_entity_set(all_entities)

    # modify the corpus and queries
    num_modified_passages = 0
    num_modified_queries = 0
    new_corpus = []
    for idx, passage in enumerate(musique_corpus):
        modified = False
        title = passage['title']
        text = passage['text']
        new_title = title
        new_text = text

        new_title = replace_phrases(new_title, entity_mapping)
        new_text = replace_phrases(new_text, entity_mapping)
        if new_title != title or new_text != text:
            num_modified_passages += 1

        new_corpus.append({'idx': passage['idx'] if 'idx' in passage else idx, 'title': new_title, 'text': new_text})

    new_dataset = []
    for sample in musique_samples:
        query = sample['question']
        new_query = replace_phrases(query, entity_mapping)
        if new_query != query:
            num_modified_queries += 1
        sample['question'] = new_query
        sample['answer'] = replace_phrases(sample['answer'], entity_mapping)
        for p in sample['paragraphs']:
            p['title'] = replace_phrases(p['title'], entity_mapping)
            p['paragraph_text'] = replace_phrases(p['paragraph_text'], entity_mapping)
        for d in sample['question_decomposition']:
            d['answer'] = replace_phrases(d['answer'], entity_mapping)
            d['question'] = replace_phrases(d['question'], entity_mapping)
        new_dataset.append(sample)

    print('Modified corpus len:', len(new_corpus), '#modified_passage:', num_modified_passages, '#modified_queries:', num_modified_queries)

    # save corpus and queries
    with open('data/musique_alternate_corpus.json', 'w') as f:
        json.dump(new_corpus, f, indent=4)
    with open('data/musique_alternate.json', 'w') as f:
        json.dump(new_dataset, f, indent=4)
