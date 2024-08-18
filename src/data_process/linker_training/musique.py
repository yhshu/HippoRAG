import argparse
import json
import os.path
import random

from django.utils.lorem_ipsum import paragraph
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from nltk import sent_tokenize
from tqdm import tqdm

from src.langchain_util import init_langchain_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='openai')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=f".dataset_{args.llm_model}.db"))
    llm = init_langchain_model(args.llm, args.llm_model)

    os.makedirs('data/linker_training/queries', exist_ok=True)
    os.makedirs('data/linker_training/corpus', exist_ok=True)
    corpus_dict = {}
    full_text_to_id = {}
    corpus_id = 0
    num_sample = {'train': 5000, 'dev': 500}
    random.seed(1)

    for split in num_sample.keys():
        path = os.path.join(args.dir, f'musique_ans_v1.0_{split}.jsonl')
        # read musique raw jsonl file
        split_data = []
        with open(path, 'r') as f:
            raw = f.readlines()
            for line in raw:
                split_data.append(json.loads(line))
        assert 0 <= num_sample[split] <= len(split_data)
        split_data = random.sample(split_data, min(num_sample[split], len(split_data)))

        # add passages to corpus
        corpus = []
        for sample in tqdm(split_data, total=len(split_data), desc=f'Processing {split}'):
            candidates = []
            for passage in sample['paragraphs']:
                # add to query data
                is_supporting = passage['is_supporting']
                full_text = passage['title'] + '\n' + passage['paragraph_text']
                if is_supporting is False:
                    continue

                candidates.append({'passage_id': full_text_to_id.get(full_text, str(corpus_id)), 'sentence': passage['paragraph_text'], 'triples': '', 'relevance': 'support'})

                if full_text in full_text_to_id:
                    continue

                # add to corpus
                corpus.append({'id': str(corpus_id), 'title': passage['title'], 'text': passage['paragraph_text'], 'full_text': full_text})
                full_text_to_id[full_text] = str(corpus_id)
                corpus_id += 1

            sample['candidates'] = candidates

        corpus_dict.update({item['id']: item for item in corpus})

        output_path = f'data/linker_training/queries/musique_ans_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')
    # end for each split

    corpus_output_path = 'data/linker_training/corpus/musique_ans_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus_dict)}) to {corpus_output_path}')
