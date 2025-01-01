import json
import os

from tqdm import tqdm

from src.data_process.create_dataset.wiki_corpus import read_enwiki_corpus
from src.pangu.retrieval_api import BM25SparseRetriever

if __name__ == '__main__':
    dataset = json.load(open('data/musique.json'))

    full_corpus, contents = read_enwiki_corpus()
    bm25_retriever = BM25SparseRetriever(contents, 'data/bm25_sparse/wiki_text')

    num_distractor = 10000
    sampled_corpus = []
    collected = set()
    for sample_idx, sample in tqdm(enumerate(dataset), 'Collecting relevant wiki text'):
        question = sample['question']
        top_k_sentences = bm25_retriever.get_top_k_sentences(question, num_distractor, True, False)
        filtered_top_k_sentences = []  # check seen set
        for p in top_k_sentences[0]:
            content = p['text']
            if content not in collected:
                collected.add(content)
                filtered_top_k_sentences.append(content)
        sampled_corpus.append(filtered_top_k_sentences)
        print(f'Sample {sample_idx} collected {len(filtered_top_k_sentences)} passages, {len(collected)} passages in total')

    all_passages = []
    for col in range(num_distractor):
        for sample_idx in range(len(sampled_corpus)):
            if col < len(sampled_corpus[sample_idx]):
                all_passages.append(sampled_corpus[sample_idx][col])
    batch_size = 10000
    batches = [all_passages[i:i + batch_size] for i in range(0, len(all_passages), batch_size)]

    os.makedirs('data/corpus_batch', exist_ok=True)
    for batch_idx, batch in enumerate(batches):
        corpus_batch = []
        for p in batch:
            p_split = p.split('\n')
            corpus_batch.append({'title': p_split[0], 'text': p_split[1]})
        with open(f'data/corpus_batch/musique_{batch_idx}_corpus.json', 'w') as f:
            json.dump(corpus_batch, f, indent=2)
            print(f'Batch {batch_idx} corpus saved to data/corpus_batch/musique_{batch_idx}_corpus.json, len: {len(corpus_batch)}')
