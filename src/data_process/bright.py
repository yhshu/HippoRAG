import copy
import json
from src.data_process.util import generate_hash
from src.pangu.retrieval_api import BM25SparseRetriever

if __name__ == '__main__':
    from datasets import load_dataset

    data = load_dataset('xlangai/BRIGHT', 'gpt4_reason')
    # load_dataset('xlangai/BRIGHT', 'examples')
    corpus_hf = load_dataset('xlangai/BRIGHT', 'documents')
    corpus_long_hf = load_dataset('xlangai/BRIGHT', 'long_documents')

    skip_subsets = ['leetcode', 'pony', 'theoremqa_theorems', 'theoremqa_questions']
    dataset = []
    corpus = []
    corpus_long = []
    passage_hash_set = set()
    passage_long_hash_set = set()

    for subset_name in data:
        if subset_name in skip_subsets:
            continue

        corpus_subset = corpus_hf[subset_name]
        corpus_subset_dict = {}
        for doc in corpus_subset:
            corpus_subset_dict[doc['id']] = doc

        corpus_long_subset = corpus_long_hf.get(subset_name, [])
        corpus_long_subset_dict = {}
        for doc in corpus_long_subset:
            corpus_long_subset_dict[doc['id']] = doc

        subset = data[subset_name]
        print("length of {}: {}".format(subset_name, len(subset)))

        corpus_subset_content = [f"{doc['id']}\n{doc['content']}" for doc in corpus_subset]
        retriever = BM25SparseRetriever(corpus_subset_content, "bright_" + subset_name)

        for sample in subset:
            dataset.append(sample)

            for gold_id_long in sample['gold_ids_long']:  # add gold long passages to corpus_long
                if gold_id_long == 'N/A':
                    continue
                passage_long_hash = generate_hash(corpus_long_subset_dict[gold_id_long]['id'] + '\n' + corpus_long_subset_dict[gold_id_long]['content'])
                if passage_long_hash in passage_long_hash_set:
                    continue
                corpus_long.append(corpus_long_subset_dict[gold_id_long])
                passage_long_hash_set.add(passage_long_hash)

            for gold_id in sample['gold_ids']:  # add gold passages to corpus
                passage_hash = generate_hash(corpus_subset_dict[gold_id]['id'] + '\n' + corpus_subset_dict[gold_id]['content'])
                if passage_hash in passage_hash_set:
                    continue
                corpus.append(corpus_subset_dict[gold_id])
                passage_hash_set.add(passage_hash)

            # use BM25 to get top passages as distractors if not in gold_ids
            query = sample['query']
            top_k_indices = retriever.get_top_k_indices(query, 10)
            for idx in top_k_indices:
                content = corpus_subset_content[idx]
                passage_hash = generate_hash(content)
                if passage_hash in passage_hash_set:
                    continue
                corpus.append(corpus_subset[idx])
                passage_hash_set.add(passage_hash)

    print()
    print("length of dataset: {}".format(len(dataset)))
    print("length of corpus: {}".format(len(corpus)))
    print("length of corpus_long: {}".format(len(corpus_long)))

    for p in corpus_long:
        if 'id' in p:
            p['title'] = p['id']
            del p['id']
        if 'content' in p:
            p['text'] = p['content']
            del p['content']

    for p in corpus:
        if 'id' in p:
            p['title'] = p['id']
            del p['id']
        if 'content' in p:
            p['text'] = p['content']
            del p['content']

    for sample in dataset:
        # rename query to question
        sample['question'] = sample.pop('query')
        sample['paragraphs'] = []
        for gold_id in sample['gold_ids']:
            for c in corpus:
                if c['title'] == gold_id:
                    c_copy = copy.deepcopy(c)
                    c_copy['is_supporting'] = True
                    sample['paragraphs'].append(c_copy)
                    break

        sample['paragraphs_long'] = []
        for gold_id_long in sample['gold_ids_long']:
            if gold_id_long == 'N/A':
                continue
            for c in corpus_long:
                if c['title'] == gold_id_long:
                    c_copy = copy.deepcopy(c)
                    c_copy['is_supporting'] = True
                    sample['paragraphs_long'].append(c_copy)
                    break

    with open('data/bright_gpt4.json', 'w') as f:
        json.dump(dataset, f)
        print("Saved to data/bright_gpt4.json")
    with open('data/bright_gpt4_corpus.json', 'w') as f:
        json.dump(corpus, f)
        print("Saved to data/bright_gpt4_corpus.json")
    with open('data/bright_gpt4_corpus_long.json', 'w') as f:
        json.dump(corpus_long, f)
        print("Saved to data/bright_gpt4_corpus_long.json")
