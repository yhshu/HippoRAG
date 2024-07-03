import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append('.')

from langchain_community.chat_models import ChatOllama

import argparse
import json
from glob import glob

import numpy as np
from langchain_openai import ChatOpenAI
from multiprocessing import Pool
from tqdm import tqdm

from src.langchain_util import init_langchain_model
from src.openie_extraction_instructions import ner_prompts, openie_post_ner_prompts
from src.processing import extract_json_dict, deduplicate_triples, fix_broken_generated_json


def print_messages(messages):
    for message in messages:
        print(message['content'])


def named_entity_recognition(passage: str, client, max_retry=5):
    ner_messages = ner_prompts.format_prompt(user_input=passage)

    done = False

    total_tokens = 0
    named_entities = []
    num_try = 0
    while not done and num_try < max_retry:
        try:
            if isinstance(client, ChatOpenAI):  # JSON mode
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0, response_format={"type": "json_object"})
                if chat_completion.response_metadata['finish_reason'] == 'length':
                    response_content = fix_broken_generated_json(chat_completion.content)
                else:
                    response_content = chat_completion.content
                response_content = eval(response_content)
                total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']
            elif isinstance(client, ChatOllama):
                response_content = client.invoke(ner_messages.to_messages())
                response_content = extract_json_dict(response_content)
                total_tokens += len(response_content.split())
            else:  # no JSON mode
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0)
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)
                total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']

            if 'named_entities' not in response_content:
                named_entities = []
            else:
                named_entities = response_content['named_entities']
            done = True
        except Exception as e:
            print('Passage NER exception', e)
        num_try += 1

    return named_entities, total_tokens


def openie_post_ner_extract(passage: str, entities: list, client):
    try:
        named_entity_json = {"named_entities": entities}
        openie_messages = openie_post_ner_prompts.format_prompt(passage=passage, named_entity_json=json.dumps(named_entity_json))

        if isinstance(client, ChatOpenAI):  # JSON mode
            chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096, response_format={"type": "json_object"})
            if chat_completion.response_metadata['finish_reason'] == 'length':
                response_content = fix_broken_generated_json(chat_completion.content)
            else:
                response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
        elif isinstance(client, ChatOllama):
            response_content = client.invoke(openie_messages.to_messages())
            response_content = extract_json_dict(response_content)
            response_content = str(response_content)
            total_tokens = len(response_content.split())
        else:  # no JSON mode
            chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096)
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)
            response_content = str(response_content)
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']

    except Exception as e:
        print('OpenIE exception', e)
        return '', 0

    return response_content, total_tokens


def extract_openie_from_triples(client, existing_json, auxiliary_file_exists, ents_by_doc, corpus_json):
    new_json = []
    all_entities = []
    chatgpt_total_tokens = 0

    for i, r in tqdm(corpus_json, total=len(corpus_json), desc='Extracting OpenIE triples'):

        passage = r['passage']

        if i < len(existing_json):
            new_json.append(existing_json[i])
        else:
            if auxiliary_file_exists:
                doc_entities = ents_by_doc[i]
            else:
                doc_entities, total_ner_tokens = named_entity_recognition(passage, client)

                doc_entities = list(np.unique(doc_entities))
                chatgpt_total_tokens += total_ner_tokens

                ents_by_doc.append(doc_entities)

            triples, total_tokens = openie_post_ner_extract(passage, doc_entities, client)

            chatgpt_total_tokens += total_tokens

            r['extracted_entities'] = doc_entities

            try:
                r['extracted_triples'] = eval(triples)["triples"]
                r['extracted_triples'] = deduplicate_triples(r['extracted_triples'])
            except Exception as e:
                print('extracting OpenIE from triples exception', e)
                print(triples)
                r['extracted_triples'] = []

            new_json.append(r)

    return (new_json, all_entities, chatgpt_total_tokens)


def openie_for_corpus(dataset_name: str, run_ner: bool, num_passages, llm: str, model_name: str, num_processes: int):
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(dataset_name, model_name, num_passages, run_ner)

    client = init_langchain_model(llm, model_name)  # LangChain model
    already_done = False
    try:
        # Get incomplete extraction output with same settings
        arg_str_regex = arg_str.replace(str(num_passages), '*')

        prev_num_passages = 0
        new_json_temp = None

        for file in glob('output/openie{}_results_{}.json'.format(dataset_name, arg_str_regex)):
            possible_json = json.load(open(file, 'r'))
            if prev_num_passages < len(possible_json['docs']):
                prev_num_passages = len(possible_json['docs'])
                new_json_temp = possible_json

        existing_json = new_json_temp['docs']
        if 'ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['ents_by_doc']
        elif 'non_dedup_ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['non_dedup_ents_by_doc']
        else:
            ents_by_doc = []

        if num_passages < len(existing_json):
            already_done = True
    except:
        existing_json = []
        ents_by_doc = []

    # Loading files which would reduce API consumption
    aux_file_str = '_'.join(flags_present) + '*_' + model_name + f'_{num_passages}'
    aux_file_str = aux_file_str.replace('{}'.format(num_passages), '*')
    auxiliary_files = glob('output/openie{}_results_{}.json'.format(dataset_name, aux_file_str))
    auxiliary_file_exists = False
    if len(auxiliary_files) > 0:
        for auxiliary_file in auxiliary_files:
            aux_info_json = json.load(open(auxiliary_file, 'r'))
            if len(aux_info_json['docs']) >= num_passages:
                ents_by_doc = aux_info_json["ents_by_doc"]
                auxiliary_file_exists = True
                print('Using Auxiliary File: {}'.format(auxiliary_file))
                break

    extracted_triples_subset = retrieval_corpus[:num_passages]
    splits = np.array_split(range(len(extracted_triples_subset)), num_processes)

    func_args = []
    for split in splits:
        func_args.append([(i, extracted_triples_subset[i]) for i in split])

    if num_processes > 1:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            packed_args = [(client, existing_json, auxiliary_file_exists, ents_by_doc, triple_json) for triple_json in func_args]
            outputs = list(executor.map(lambda args: extract_openie_from_triples(*args), packed_args))
    else:
        outputs = [extract_openie_from_triples(client, existing_json, auxiliary_file_exists, ents_by_doc, corpus_json) for corpus_json in func_args]

    new_json = []
    all_entities = []
    lm_total_tokens = 0

    for output in outputs:
        new_json.extend(output[0])
        all_entities.extend(output[1])
        lm_total_tokens += output[2]

    if not (already_done):
        avg_ent_chars = np.mean([len(e) for e in all_entities])
        avg_ent_words = np.mean([len(e.split()) for e in all_entities])

        # Current Cost
        approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

        extra_info_json = {"docs": new_json,
                           "ents_by_doc": ents_by_doc,
                           "avg_ent_chars": avg_ent_chars,
                           "avg_ent_words": avg_ent_words,
                           "num_tokens": lm_total_tokens,
                           "approx_total_tokens": approx_total_tokens,
                           }
        output_path = 'output/openie{}_results_{}.json'.format(dataset_name, arg_str)
        json.dump(extra_info_json, open(output_path, 'w'))
        print('OpenIE saved to', output_path)


def load_corpus(dataset_name: str, model_name: str, num_passages, run_ner):
    corpus = json.load(open(f'data/{dataset_name}_corpus.json', 'r'))
    if 'hotpotqa' in dataset_name:
        keys = list(corpus.keys())
        retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(corpus[key])} for i, key in enumerate(keys)]
    else:
        retrieval_corpus = corpus
        for document in retrieval_corpus:
            document['passage'] = document['title'] + '\n' + document['text']
    dataset_name = '_' + dataset_name
    if num_passages == 'all':
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except:
            assert False, "Set 'num_passages' to an integer or 'all'"
    flag_names = ['ner']
    flags_present = [flag_names[i] for i, flag in enumerate([run_ner]) if flag]
    if len(flags_present) > 0:
        arg_str = '_'.join(flags_present) + '_' + model_name.replace('/', '_') + f'_{num_passages}'
    else:
        arg_str = model_name.replace('/', '_') + f'_{num_passages}'
    print(arg_str)
    return arg_str, dataset_name, flags_present, num_passages, retrieval_corpus


def openie_for_corpus_openai_batch(dataset_name: str, run_ner: bool, num_passages, model_name: str):
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(dataset_name, model_name, num_passages, run_ner)

    # output corpus to a file to upload to OpenAI
    corpus_jsonl_path = f'output/{arg_str}.jsonl'
    jsonl_contents = []
    for i, passage in enumerate(retrieval_corpus):
        jsonl_contents.append(json.dumps({"custom_id": i, "method": "POST", "url": "/v1/chat/completions", "messages": [],
                                          "max_tokens": 4096}))
    with open(corpus_jsonl_path, 'w') as f:
        f.write('\n'.join(jsonl_contents))

    from openai import OpenAI
    client = OpenAI()
    batch_input_file = client.files.create(file=open(corpus_jsonl_path, 'rb'), purpose='batch')
    # todo to complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='10')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=10)
    parser.add_argument('--openai_batch', action='store_true', help='Use OpenAI batch API, if this is set, `num_processes`, `llm` are ignored.')

    args = parser.parse_args()
    if args.openai_batch:
        openie_for_corpus_openai_batch(args.dataset, args.run_ner, args.num_passages, args.model_name)
    else:
        openie_for_corpus(args.dataset, args.run_ner, args.num_passages, args.llm, args.model_name, args.num_processes)
