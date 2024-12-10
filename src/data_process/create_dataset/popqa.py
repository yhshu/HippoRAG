import argparse
import sys

import requests

sys.path.append('.')

import json
import os
import random

import pandas as pd
from tqdm import tqdm

from src.data_process.util import generate_hash
from src.pangu.retrieval_api import BM25SparseRetriever
from src.processing import query_data_has_duplication, corpus_has_duplication


def get_english_wikipedia_link(wikidata_id):
    """
    Get the English Wikipedia link for a given Wikidata ID.

    :param wikidata_id: The Wikidata ID (e.g., "Q42" for Douglas Adams).
    :return: English Wikipedia URL or None if not found.
    """
    # Wikidata API endpoint
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbgetentities',  # API action to get entity details
        'ids': wikidata_id,  # Wikidata ID
        'props': 'sitelinks',  # Include sitelinks in response
        'format': 'json'  # JSON format
    }

    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Parse the JSON response
        data = response.json()
        sitelinks = data.get('entities', {}).get(wikidata_id, {}).get('sitelinks', {})

        # Extract the English Wikipedia link
        enwiki = sitelinks.get('enwiki', {})
        wikipedia_url = enwiki.get('url')  # URL field in the sitelink entry
        return wikipedia_url
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_wikipedia_text(page_title, lang='en'):
    """
    Fetch plain text content of a Wikipedia page by its title.

    :param page_title: The title of the Wikipedia page (e.g., "Douglas_Adams").
    :param lang: Language code for Wikipedia (e.g., "en", "zh").
    :return: Text content of the page or None if not found.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'prop': 'extracts',
        'explaintext': True,
        'titles': page_title,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page:
            return page["extract"]
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki', type=str, default='/fs/project/PAS1576/yiheng/workspace/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl')
    args = parser.parse_args()

    df = pd.read_csv('data/popQA.tsv', sep='\t')

    data = []
    for row in tqdm(df.iterrows(), desc='Reading data...'):
        data.append(json.loads(row[1].to_json()))

    random.seed(1)

    full_corpus = []
    contents = []
    content_hash_set = set()
    with open(args.wiki) as f:
        for line in tqdm(f, 'Processing wiki text'):
            item = json.loads(line.strip())
            title = item['title'] + ' - ' + item['section']
            text = item['text']
            content = title + '\n' + text
            content_hash = generate_hash(content)
            if content_hash not in content_hash_set:
                content_hash_set.add(content_hash)
                full_corpus.append({'idx': len(full_corpus), 'title': title, 'text': text})
                contents.append(content)

    os.makedirs('data/bm25_sparse/wiki_text', exist_ok=True)
    bm25_retriever = BM25SparseRetriever(contents, 'data/bm25_sparse/wiki_text')

    corpus = []
    corpus_content_hash_set = set()
    new_data = []
    content_hash_set = set()

    for sample in tqdm(data, 'Collecting relevant wiki text'):
        subject_wikipedia_text = get_wikipedia_text(sample['s_wiki_title'])
        object_wikipedia_text = get_wikipedia_text(sample['o_wiki_title'])

        if subject_wikipedia_text is None or object_wikipedia_text is None:
            continue
        if subject_wikipedia_text == '' or object_wikipedia_text == '':
            continue

        subject_first_paragraph = subject_wikipedia_text.split('\n\n\n== ')
        object_first_paragraph = object_wikipedia_text.split('\n\n\n== ')

        content_hash = generate_hash(subject_first_paragraph[0].strip())
        if content_hash not in corpus_content_hash_set:
            corpus_content_hash_set.add(content_hash)
            corpus.append({'title': sample['s_wiki_title'], 'text': subject_first_paragraph[0]})

        content_hash = generate_hash(object_first_paragraph[0].strip())
        if content_hash not in corpus_content_hash_set:
            corpus_content_hash_set.add(content_hash)
            corpus.append({'title': sample['o_wiki_title'], 'text': object_first_paragraph[0]})

        sample['paragraphs'] = []
        sample['paragraphs'].append({'title': sample['s_wiki_title'], 'text': subject_first_paragraph[0], 'is_supporting': True})
        sample['paragraphs'].append({'title': sample['o_wiki_title'], 'text': object_first_paragraph[0], 'is_supporting': True})

        k = 5
        indices = set()
        top_indices = bm25_retriever.get_top_k_indices(sample['question'], k, True, False)
        indices.update(top_indices)
        top_indices = bm25_retriever.get_top_k_indices(sample['o_wiki_title'], k, True, False)
        indices.update(top_indices)

        assert  k <= len(indices) <= 2 * k
        for idx in indices:
            content = full_corpus[idx]['text'].strip()
            content_hash = generate_hash(content)
            if content_hash not in corpus_content_hash_set:
                corpus_content_hash_set.add(content_hash)
                corpus.append(full_corpus[idx])

        new_data.append(sample)
        if len(new_data) >= 1000:
            break

    corpus_output_path = 'data/popqa_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f)
        print(f'{len(corpus)} passages saved to {corpus_output_path}')

    dataset_output_path = 'data/popqa.json'
    with open(dataset_output_path, 'w') as f:
        json.dump(new_data, f)
        print(f'{len(new_data)} samples saved to {dataset_output_path}')

    query_data_has_duplication(new_data, False, False)
    corpus_has_duplication(corpus)
