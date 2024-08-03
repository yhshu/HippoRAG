import copy
import json
import random
import re

from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

from src.data_process.util import chunk_corpus
from src.langchain_util import num_tokens_by_tiktoken


def process_html_to_raw_text(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    return text


if __name__ == '__main__':
    dataset = load_dataset("deepmind/narrativeqa")
    print(dataset.shape)

    documents = {}
    num_tokens = 0

    for sample in tqdm(dataset['validation']):
        document = sample['document']['text']
        documents[sample['document']['id']] = document

    print('num doc', len(documents))
    for doc in documents.values():
        num_tokens += num_tokens_by_tiktoken(doc)
    print(round(num_tokens / len(documents), 3))

    # randomly sample 10 documents to get a new dict
    sampled_doc_ids = random.sample(list(documents.keys()), 10)
    sample_docs = {k: documents[k] for k in sampled_doc_ids}

    corpus = {}
    data = []
    for sample in tqdm(dataset['validation']):
        s = copy.deepcopy(sample)
        if s['document']['id'] in sample_docs:
            s['question'] = sample['question']['text']
            s['answer'] = [a['text'] for a in sample['answers']]
            assert 'question' in s and 'answer' in s
            del s['answers']

            data.append(s)
            html_text = sample['document']['text']
            doc_text = process_html_to_raw_text(html_text)
            doc_text_split = doc_text.split('\n')
            first_lines = '\n'.join(doc_text_split[:1])
            main_text = '\n'.join(doc_text_split[1:])
            corpus[s['document']['id']] = {'idx': s['document']['id'], 'title': first_lines, 'text': main_text}

    dataset_name = 'narrativeqa_dev_10_doc'
    data_output_path = f'data/{dataset_name}.json'
    corpus_output_path = f'data/{dataset_name}_corpus.json'

    with open(data_output_path, 'w') as f:
        json.dump(data, f, indent=4)

    corpus = [corpus[k] for k in corpus]
    corpus = chunk_corpus(corpus, chunk_size=128)
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f, indent=4)
