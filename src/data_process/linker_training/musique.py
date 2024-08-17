import argparse
import json
import os.path
import random

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.langchain_util import init_langchain_model

system_message = """Given a query and a supporting passage for this query, the task is:
1. Extract a piece from the passage that helps reasoning or answering this query.
2. Extract triples from the above generated sentence in the form of (subject, predicate, object). 

Requirements:

- The query may require multi-hop reasoning, so this passage may only be one of the supporting passages for the query. 
- Respond with the sentence and a fact list in JSON format."""

demo_input = """Query: What is the capital of the country where the Eiffel Tower is located?

Passage: The Eiffel Tower is located in Paris, which is the capital of France. France is a country in Europe. The Eiffel Tower was built in 1887.
"""

demo_output = """{
    "sentence": "The Eiffel Tower is located in Paris, which is the capital of France.",
    "triples": [
        [
            "Eiffel Tower",
            "location",
            "Paris"
        ],
        [
            "France",
            "capital",
            "Paris"
        ]
    ]
}"""


def get_supporting_sentence_from_passage(query: str, passage: str, llm):
    user_prompt = f"Query: {query}\n\nPassage: {passage}"
    messages = [SystemMessage(system_message), HumanMessage(demo_input), AIMessage(demo_output), HumanMessage(user_prompt)]
    completions = llm.invoke(messages, temperature=0.0, response_format={"type": "json_object"})
    try:
        content = completions.content
        response = json.loads(content)
        sentence = response['sentence']
        triples = response['triples']
        return sentence, triples
    except Exception as e:
        print(f'Error: {e}')
        return '', []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='openai')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    args = parser.parse_args()

    set_llm_cache(SQLiteCache(database_path=f".dataset_{args.llm_model}.db"))
    llm = init_langchain_model(args.llm, args.llm_model)

    corpus_dict = {}
    full_text_set = set()
    corpus_id = 0
    num_sample = {'train': 1000, 'dev': 200}
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
        for sample in split_data:
            candidates = []
            for passage in sample['paragraphs']:
                # add to query data
                is_supporting = passage['is_supporting']
                if is_supporting:
                    sentence, triples = get_supporting_sentence_from_passage(sample['question'], passage['paragraph_text'], llm)
                    candidates.append({'passage_id': str(corpus_id), 'sentence': sentence, 'triples': triples, 'relevance': 'support'})

                full_text = passage['title'] + '\n' + passage['paragraph_text']
                if full_text in full_text_set:
                    continue

                # add to corpus
                corpus.append({'id': str(corpus_id), 'title': passage['title'], 'text': passage['paragraph_text'], 'full_text': full_text})
                corpus_id += 1
                full_text_set.add(full_text)

            sample['candidates'] = candidates

        corpus_dict = {item['id']: item for item in corpus}

        os.makedirs('data/linker_training', exist_ok=True)
        output_path = f'data/linker_training/musique_ans_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')

    corpus_output_path = 'data/linker_training/musique_ans_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus_dict, f)
    print(f'Saving corpus ({len(corpus)}) to {corpus_output_path}')
