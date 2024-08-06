import argparse
import json
import os
from typing import Dict

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class KB():
    summaries: Dict[str, str]

    def __init__(self):
        self.entities = {}
        self.triples = []
        self.summaries = {}
        self.title_page = {}

    def are_triples_equal(self, t1, t2):
        return all(t1[attr] == t2[attr] for attr in ['head', 'relation', 'tail'])

    def exists_triple(self, t1):
        return any(self.are_triples_equal(t1, t2) for t2 in self.triples)

    def merge_triples(self, t1):
        if 'meta' not in t1:
            return
        t2 = [t for t in self.triples
              if self.are_triples_equal(t1, t)][0]
        spans_to_add = [span for span in t1["meta"]["spans"]
                        if span not in t2["meta"]["spans"]]
        t2["meta"]["spans"] += spans_to_add

    def add_entity(self, e):
        self.entities[e["title"]] = {k: v for k, v in e.items() if k != "title"}

    def add_triple(self, triple):
        # check on wikipedia
        candidate_entities = [triple["head"], triple["tail"]]
        entities = [{'title': ent} for ent in candidate_entities]

        # if no entity exists, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        triple["head"] = entities[0]
        triple["tail"] = entities[1]

        # manage new relation
        if not self.exists_triple(triple):
            self.triples.append(triple)
        else:
            self.merge_triples(triple)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Triples:")
        for t in self.triples:
            print(f"  {t}")


def extract_triplets(response):
    triples = []
    for triple in response.triples:
        triples.append((triple['head']['title'], triple['relation'], triple['tail']['title']))
    return triples


def extract_triples_from_model_output(text):
    triples = []
    subject, relation, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triples.append({
                    'head': subject.strip(),
                    'relation': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triples.append({
                    'head': subject.strip(),
                    'relation': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triples.append({
            'head': subject.strip(),
            'relation': relation.strip(),
            'tail': object_.strip()
        })
    return triples


def from_text_to_kb(texts, model, tokenizer, device='cuda'):
    # Check if the model is an instance of nn.DataParallel
    if isinstance(model, nn.DataParallel):
        # If it is, access the original model
        model = model.module

    num_return_sequences = 3
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        length_penalty=0,
        num_beams=3,
        num_return_sequences=num_return_sequences
    )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    kbs = []
    for i in range(len(texts)):
        kb = KB()
        predicted_sentences = generated_texts[i * num_return_sequences: (i + 1) * num_return_sequences]
        for sentence in predicted_sentences:
            triples = extract_triples_from_model_output(sentence)
            for triple in triples:
                kb.add_triple(triple)
        kbs.append(kb)
    return kbs


class CorpusDataset(Dataset):
    def __init__(self, corpus):
        self.items = corpus

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return idx, self.items[idx]['title'] + '\n' + self.items[idx]['text']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--start', type=int, help='start index', default=0)
    parser.add_argument('--end', type=int, help='end index', default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    assert args.end is None or args.end <= len(corpus)
    assert args.start is None or 0 <= args.start < len(corpus)
    corpus = corpus[args.start:args.end]

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(args.device)

    os.makedirs(f'output/rebel', exist_ok=True)
    triple_output_path = f'output/rebel/{args.dataset}_rebel_extracted_triples.json'
    entity_output_path = f'output/rebel/{args.dataset}_rebel_extracted_entities.json'
    relation_output_path = f'output/rebel/{args.dataset}_rebel_extracted_relations.json'

    all_triples = []
    all_entities = set()
    all_relations = set()

    corpus_dataset = CorpusDataset(corpus)
    dataloader = DataLoader(corpus_dataset, batch_size=48, num_workers=1)

    # use dataloader to call model with batch and multi-gpu
    for passage_idx, passage in tqdm(dataloader, desc='Extraction'):

        # extract the passage to triples
        try:
            responses = from_text_to_kb(passage, model, tokenizer)
        except Exception as e:
            print(e)
            print('Failed to extract triples', e)
            continue

        # store the triples
        triples = []
        try:
            for r in responses:
                t = extract_triplets(r)
                triples.append(t)
        except Exception as e:
            print(e)
            print(responses)
        # print(triples)
        else:
            for t in triples:  # for each passage
                for triple in t:  # for each triple
                    triple = list(triple)
                    if len(triple) != 3:
                        continue
                    if isinstance(triple[0], str) and any(char.isalpha() for char in triple[0]):
                        all_entities.add(triple[0])
                    all_relations.add(triple[1])
                    if isinstance(triple[2], str) and any(char.isalpha() for char in triple[2]):
                        all_entities.add(triple[2])

            for seq_id, idx in enumerate(list(passage_idx)):
                idx = int(idx)
                all_triples.append({'idx': idx, 'passage': passage[seq_id], 'triples': triples[seq_id]})
            if any((tensor.remainder(50) == 0).any() for tensor in passage_idx):
                json.dump(all_triples, open(triple_output_path, 'w'))
                json.dump(list(all_entities), open(entity_output_path, 'w'))
                json.dump(list(all_relations), open(relation_output_path, 'w'))
    # end for each passage

    json.dump(all_triples, open(triple_output_path, 'w'))
    json.dump(list(all_entities), open(entity_output_path, 'w'))
    json.dump(list(all_relations), open(relation_output_path, 'w'))
