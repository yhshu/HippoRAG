import sys

sys.path.append('.')

import vllm
from concurrent.futures import ThreadPoolExecutor

import argparse
import json
from glob import glob

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_community.chat_models import ChatOllama
from tqdm import tqdm

from src.langchain_util import init_langchain_model
from src.openie_extraction_instructions import ner_prompts, openie_post_ner_prompts
from src.processing import extract_json_dict, deduplicate_triples, fix_broken_generated_json, corpus_has_duplication


def print_messages(messages):
    for message in messages:
        print(message['content'])


def named_entity_recognition(passage: str, client, max_retry=5, extractor_name=None):
    if isinstance(client, vllm.LLM):
        return named_entity_recognition_batch_vllm(passage, client, max_retry, extractor_name)
    ner_messages = ner_prompts.format_prompt(user_input=passage)

    done = False

    total_tokens = 0
    named_entities = []
    num_try = 0
    while not done and num_try < max_retry:
        try:
            if isinstance(client, ChatOpenAI):  # JSON mode
                completion = client.invoke(ner_messages.to_messages(), temperature=0, response_format={"type": "json_object"})
                if completion.response_metadata['finish_reason'] == 'length':
                    response_content = fix_broken_generated_json(completion.content)
                else:
                    response_content = completion.content
                response_content = eval(response_content)
                total_tokens += completion.response_metadata['token_usage']['total_tokens']
            elif isinstance(client, ChatOllama):
                response_content = client.invoke(ner_messages.to_messages())
                response_content = extract_json_dict(response_content)
                total_tokens += len(response_content.split())
            elif isinstance(client, VLLMOpenAI):
                from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt
                if client.model_name.startswith('meta-llama/Llama-3'):
                    prompt = langchain_message_to_llama_3_prompt(ner_messages.to_messages())
                else:
                    prompt = ner_messages.to_string()
                extra_body = {"response_format": {"type": "json_object"}}
                completion = client.invoke(prompt, extra_body=extra_body)
                response_content = extract_json_dict(completion)
                total_tokens += len(completion.split())
            else:  # no JSON mode
                completion = client.invoke(ner_messages.to_messages(), temperature=0)
                response_content = completion.content
                response_content = extract_json_dict(response_content)
                total_tokens += completion.response_metadata['token_usage']['total_tokens']

            if 'named_entities' not in response_content:
                named_entities = []
            else:
                named_entities = response_content['named_entities']
            done = True
        except Exception as e:
            print('Passage NER exception:', e)
        num_try += 1

    return named_entities, total_tokens


def openie_post_ner_extract(passage: str, entities: list, client, extractor_name=None):
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
        elif isinstance(client, vllm.LLM):
            from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt
            if extractor_name.startswith('meta-llama/Llama-3'):
                prompt = langchain_message_to_llama_3_prompt(openie_messages.to_messages())
            else:
                prompt = openie_messages.to_string()
            from outlines.serve.vllm import JSONLogitsProcessor
            from vllm import SamplingParams
            logits_processor = JSONLogitsProcessor(OpenIEModel.model_json_schema(), client)
            completion = client.generate(prompt,
                                         sampling_params=SamplingParams(max_tokens=512, temperature=0,
                                                                        logits_processors=[logits_processor]))
            response_content = completion[0].outputs[0].text
            total_tokens = len(completion[0].outputs[0].token_ids)
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


def named_entity_recognition_batch_vllm(client, passages, extractor_name=None):
    from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt, PROMPT_JSON_TEMPLATE
    from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest
    from vllm import SamplingParams

    assert isinstance(client, vllm.LLM)
    all_prompts = [ner_prompts.format_prompt(user_input=passage) for passage in passages]
    if extractor_name.startswith('meta-llama/Llama-3'):
        all_prompts = [langchain_message_to_llama_3_prompt(prompt.to_messages()) for prompt in all_prompts]
    else:
        all_prompts = [prompt.to_string() for prompt in all_prompts]

    print(all_prompts[0])
    vllm_output = client.generate(
        all_prompts,
        sampling_params=SamplingParams(max_tokens=512, temperature=0),
        guided_options_request=GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE['ner'])
    )
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    all_responses = [extract_json_dict(response) for response in all_responses]
    all_total_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
    return all_responses, all_total_tokens


def openie_post_ner_extract_batch_vllm(client, passages, entities_list, extractor_name=None):
    assert isinstance(client, vllm.LLM)
    named_entity_json_list = [{"named_entities": entities} for entities in entities_list]
    openie_messages = [openie_post_ner_prompts.format_prompt(passage=passage, named_entity_json=json.dumps(named_entity_json)) for passage, named_entity_json in
                       zip(passages, named_entity_json_list)]
    from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt, PROMPT_JSON_TEMPLATE
    if extractor_name.startswith('meta-llama/Llama-3'):
        all_prompts = [langchain_message_to_llama_3_prompt(prompt.to_messages()) for prompt in openie_messages]
    else:
        all_prompts = [prompt.to_string() for prompt in openie_messages]

    from vllm import SamplingParams
    from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest

    vllm_output = client.generate(
        all_prompts,
        sampling_params=SamplingParams(max_tokens=512, temperature=0),
        guided_options_request=GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE['triples'])
    )
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    all_total_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
    return all_responses, all_total_tokens


def extract_openie_from_triples_batch_vllm(client, existing_json, auxiliary_file_exists, ents_by_doc, corpus_json, extractor_name=None):
    assert isinstance(client, vllm.LLM)
    extractions = []
    llm_total_tokens = 0
    missing_entity_doc_ids, missing_entity_passages = [], []
    all_entities = []
    post_ner_passages, post_ner_entities = [], []
    for i, sample in corpus_json:
        if i < len(existing_json):
            extractions.append(existing_json[i])
            all_entities.extend(existing_json[i]['extracted_entities'])
        else:
            post_ner_passages.append(sample['passage'])
            if auxiliary_file_exists:
                all_entities.extend(ents_by_doc[i])
                sample['extracted_entities'] = ents_by_doc[i]
                post_ner_entities.append(ents_by_doc[i])
            else:
                missing_entity_doc_ids.append(i)
                missing_entity_passages.append(sample['passage'])
                ents_by_doc.append([])
                sample['extracted_entities'] = []

    if len(missing_entity_doc_ids) > 0:
        # do NER in batch
        all_responses, all_total_tokens = named_entity_recognition_batch_vllm(client, missing_entity_passages, extractor_name)
        llm_total_tokens += sum(all_total_tokens)
        for id, response in zip(missing_entity_doc_ids, all_responses):
            doc_entities = response['named_entities']
            doc_entities = list(np.unique(doc_entities))
            ents_by_doc[id] = doc_entities
            corpus_json[id][1]['extracted_entities'] = doc_entities
            all_entities.extend(doc_entities)
            post_ner_entities.append(doc_entities)

    assert len(post_ner_passages) == len(post_ner_entities)
    if len(post_ner_passages) > 0:
        all_responses, all_total_tokens = openie_post_ner_extract_batch_vllm(client, post_ner_passages, post_ner_entities, extractor_name)
        llm_total_tokens += sum(all_total_tokens)
        for response, sample_t in zip(all_responses, corpus_json):
            sample = sample_t[1]
            try:
                if response == '':
                    sample['extracted_triples'] = []
                    print('Got empty triples from openie_post_ner_extract')
                else:
                    sample['extracted_triples'] = eval(response)["triples"]
                    sample['extracted_triples'] = deduplicate_triples(sample['extracted_triples'])
            except Exception as e:
                print('extracting OpenIE from triples exception', e)
                print(response)
                sample['extracted_triples'] = []
            extractions.append(sample)
    return (extractions, all_entities, llm_total_tokens)


def extract_openie_from_triples(client, existing_json, auxiliary_file_exists, ents_by_doc, corpus_json,
                                extractor_name=None):
    if isinstance(client, vllm.LLM):
        return extract_openie_from_triples_batch_vllm(client, existing_json, auxiliary_file_exists, ents_by_doc, corpus_json, extractor_name)
    extractions = []
    all_entities = []
    llm_total_tokens = 0

    for i, sample in tqdm(corpus_json, total=len(corpus_json), desc='Extracting OpenIE triples'):

        passage = sample['passage']

        if i < len(existing_json):
            extractions.append(existing_json[i])
        else:
            if auxiliary_file_exists:
                doc_entities = ents_by_doc[i]
            else:
                doc_entities, total_ner_tokens = named_entity_recognition(passage, client, extractor_name=extractor_name)

                doc_entities = list(np.unique(doc_entities))
                llm_total_tokens += total_ner_tokens

                ents_by_doc.append(doc_entities)

            triples, total_tokens = openie_post_ner_extract(passage, doc_entities, client, extractor_name)

            llm_total_tokens += total_tokens

            sample['extracted_entities'] = doc_entities

            try:
                if triples == '':
                    sample['extracted_triples'] = []
                    print('Got empty triples from openie_post_ner_extract')
                else:
                    sample['extracted_triples'] = eval(triples)["triples"]
                    sample['extracted_triples'] = deduplicate_triples(sample['extracted_triples'])
            except Exception as e:
                print('extracting OpenIE from triples exception', e)
                print(triples)
                sample['extracted_triples'] = []

            extractions.append(sample)
            all_entities.extend(doc_entities)

    return (extractions, all_entities, llm_total_tokens)


def openie_for_corpus(dataset_name: str, run_ner: bool, num_passages, llm: str, model_name: str, num_processes: int, num_gpus: int = 4):
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(dataset_name, model_name, num_passages, run_ner)

    client = init_langchain_model(llm, model_name, num_gpus=num_gpus)  # LangChain model
    already_done = False
    try:
        # Get incomplete extraction output with same settings
        arg_str_regex = arg_str.replace(str(num_passages), '*')

        prev_num_passages = 0
        extraction_json_temp = None

        for file in glob('output/openie{}_results_{}.json'.format(dataset_name, arg_str_regex)):
            possible_json = json.load(open(file, 'r'))
            if prev_num_passages < len(possible_json['docs']):
                prev_num_passages = len(possible_json['docs'])
                extraction_json_temp = possible_json

        existing_json = extraction_json_temp['docs']
        if 'ents_by_doc' in extraction_json_temp:
            ents_by_doc = extraction_json_temp['ents_by_doc']
        elif 'non_dedup_ents_by_doc' in extraction_json_temp:
            ents_by_doc = extraction_json_temp['non_dedup_ents_by_doc']
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
            packed_args = [(client, existing_json, auxiliary_file_exists, ents_by_doc, triple_json, model_name)
                           for triple_json in func_args]
            outputs = list(executor.map(lambda args: extract_openie_from_triples(*args), packed_args))
    else:
        outputs = [extract_openie_from_triples(client, existing_json, auxiliary_file_exists,
                                               ents_by_doc, corpus_json, model_name)
                   for corpus_json in func_args]

    extraction_by_doc = []
    all_entities = []
    lm_total_tokens = 0

    for output in outputs:
        extraction_by_doc.extend(output[0])
        all_entities.extend(output[1])
        lm_total_tokens += output[2]

    if not (already_done):
        avg_ent_chars = np.mean([len(e) for e in all_entities])
        avg_ent_words = np.mean([len(e.split()) for e in all_entities])

        # Current Cost
        approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

        extra_info_json = {"docs": extraction_by_doc,
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
    model_name_processed = model_name.replace('/', '_')
    corpus = json.load(open(f'data/{dataset_name}_corpus.json', 'r'))
    assert corpus_has_duplication(corpus) is False
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
        arg_str = '_'.join(flags_present) + '_' + model_name_processed + f'_{num_passages}'
    else:
        arg_str = model_name_processed + f'_{num_passages}'
    print(arg_str)
    return arg_str, dataset_name, flags_present, num_passages, retrieval_corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='all')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=10)

    args = parser.parse_args()
    openie_for_corpus(args.dataset, args.run_ner, args.num_passages, args.llm, args.model_name, args.num_processes)
