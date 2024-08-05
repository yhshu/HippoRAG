import sys

sys.path.append('.')

import argparse
import json

from openai import OpenAI
from tqdm import tqdm

from src.openie_with_retrieval_option_parallel import load_corpus


def fix_broken_triple_json(input_str):
    last_index = input_str.rfind("],")
    if last_index != -1:
        result_str = input_str[:last_index + 2] + "}"
    else:
        result_str = input_str
    return result_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--file_id', type=str, help="OpenAI file ID to retrieve", required=True)
    parser.add_argument('--num_passages', type=str, default='all')
    args = parser.parse_args()

    client = OpenAI(max_retries=5, timeout=60)
    content = client.files.content(file_id=args.file_id)

    # save file to do OpenIE after NER
    lines = content.text.strip().split("\n")
    print('content length:', len(lines))

    run_ner = True
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(args.dataset, args.model_name, args.num_passages, run_ner)

    response = {}  # custom id -> response, note that batch API doesn't guarantee order
    for line in tqdm(lines):
        item = json.loads(line)
        custom_id = item['custom_id']
        response[custom_id] = item['response']['body']['choices'][0]['message']['content']

    extraction_by_doc = []
    ents_by_doc = []
    avg_ent_chars = 0
    avg_ent_words = 0
    num_entities = 0

    for i, passage in enumerate(retrieval_corpus):
        idx = str(passage['idx']) if 'idx' in passage else str(i)
        try:
            extraction = json.loads(response[idx]).get('triples', [])
        except:
            try:
                extraction = json.loads(fix_broken_triple_json(response[idx])).get('triples', [])
            except:
                extraction = []
                print('Error loading extraction response')

        entities = set()
        triples = []
        for e in extraction:
            if isinstance(e, list) and len(e) > 0 and isinstance(e[0], str):
                if len(e) and isinstance(e[0], str):
                    entities.add(e[0])
                if len(e) == 3 and isinstance(e[2], str):
                    entities.add(e[2])
                    triples.append(e)
            else:
                print('Wrong type in extraction:', type(extraction))

        item = {'idx': idx, 'title': passage['title'], 'text': passage['text'], 'passage': passage['passage'],
                'extracted_entities': list(entities), 'extracted_triples': triples}
        extraction_by_doc.append(item)
        ents_by_doc.append(list(entities))
        num_entities += len(entities)
        avg_ent_chars += sum([len(str(ent)) for ent in entities])
        avg_ent_words += sum([len(str(ent).split()) for ent in entities])

    avg_ent_chars /= num_entities
    avg_ent_words /= num_entities

    extra_info_json = {"docs": extraction_by_doc,
                       "ents_by_doc": ents_by_doc,
                       "avg_ent_chars": avg_ent_chars,
                       "avg_ent_words": avg_ent_words
                       }
    output_path = 'output/openie{}_results_{}.json'.format(dataset_name, arg_str)
    json.dump(extra_info_json, open(output_path, 'w'))
    print('OpenIE saved to', output_path, 'len:', len(extraction_by_doc))
