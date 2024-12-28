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

def process_file(file_id, client):
    content = client.files.content(file_id=file_id)
    lines = content.text.strip().split("\n")
    print(f'Processing file {file_id}, content length:', len(lines))

    response = {}  # custom_id -> response, adjusted for global custom IDs
    for i, line in tqdm(enumerate(lines)):
        item = json.loads(line)
        response[item['custom_id']] = item['response']['body']['choices'][0]['message']['content']

    return response, len(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--file_ids', type=str, nargs='+', help="List of OpenAI file IDs to retrieve", required=True)
    parser.add_argument('--num_passages', type=str, default='all')
    args = parser.parse_args()

    client = OpenAI(max_retries=5, timeout=60)

    all_responses = {}
    all_extraction_by_doc = []
    all_ents_by_doc = []
    num_entities = 0
    avg_ent_chars = 0
    avg_ent_words = 0

    run_ner = True
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(
        args.dataset, args.model_name, args.num_passages, run_ner)

    # Process each file and accumulate results
    for file_id in args.file_ids:
        responses, num_lines = process_file(file_id, client)
        all_responses.update(responses)

    for i, passage in enumerate(retrieval_corpus):
        response = all_responses.get(f"global_{i}", "")
        if response == "":
            print(f'Idx {i}: no response found')
        try:
            extraction = json.loads(response).get('triples', [])
        except:
            try:
                extraction = json.loads(fix_broken_triple_json(response)).get('triples', [])
            except:
                extraction = []
                print(f'Idx {i}, error when loading extraction response')

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
                print(f'Idx {i}, wrong type in extraction: {type(extraction)}')

        item = {'idx': i, 'title': passage['title'], 'text': passage['text'], 'passage': passage['passage'],
                'extracted_entities': list(entities), 'extracted_triples': triples}
        all_extraction_by_doc.append(item)
        all_ents_by_doc.append(list(entities))
        num_entities += len(entities)
        avg_ent_chars += sum([len(str(ent)) for ent in entities])
        avg_ent_words += sum([len(str(ent).split()) for ent in entities])

    avg_ent_chars /= num_entities
    avg_ent_words /= num_entities

    # Final output
    extra_info_json = {"docs": all_extraction_by_doc,
                       "ents_by_doc": all_ents_by_doc,
                       "avg_ent_chars": avg_ent_chars,
                       "avg_ent_words": avg_ent_words
                       }
    output_path = 'output/openie{}_results_{}.json'.format(dataset_name, arg_str)
    json.dump(extra_info_json, open(output_path, 'w'))
    print('OpenIE saved to', output_path, 'len:', len(all_extraction_by_doc))

if __name__ == '__main__':
    main()
