import sys

sys.path.append('.')

from functools import partial
from src.processing import extract_json_dict
from langchain_community.chat_models import ChatOllama

sys.path.append('.')
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from tqdm import tqdm

from src.langchain_util import init_langchain_model
from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt, PROMPT_JSON_TEMPLATE

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""


def named_entity_recognition(client, text: str):
    """
    Named entity recognition
    @param client:
    @param text:
    @return: a dict {"named_entities": a list of named entities}, an integer total_tokens
    """
    query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
                                                          HumanMessage(query_prompt_one_shot_input),
                                                          AIMessage(query_prompt_one_shot_output),
                                                          HumanMessage(query_prompt_template.format(text))])
    query_ner_messages = query_ner_prompts.format_prompt()

    json_mode = False
    if isinstance(client, ChatOpenAI):  # JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
        response_content = chat_completion.content
        total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
        json_mode = True
    elif isinstance(client, ChatOllama):
        response_content = client.invoke(query_ner_messages.to_messages())
        response_content = extract_json_dict(response_content)
        total_tokens = len(response_content.split())
    else:  # no JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'])
        response_content = chat_completion.content
        response_content = extract_json_dict(response_content)
        total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']

    if not json_mode:
        try:
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except Exception as e:
            print('Query NER exception', e)
            response_content = {'named_entities': []}

    return response_content, total_tokens


def run_ner_on_texts(client, texts: list):
    """
    Named entity recognition on a list of texts
    @param client:
    @param texts:
    @return:
    """
    ner_output = []
    total_tokens = 0

    for text in tqdm(texts):
        ner, num_token = named_entity_recognition(client, text)
        ner_output.append(ner)
        total_tokens += num_token

    return ner_output, total_tokens


def run_ner_on_texts_vllm(client, all_queries):
    import vllm
    assert isinstance(client, vllm.LLM)
    query_ner_prompts = [
        ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
                                          HumanMessage(query_prompt_one_shot_input),
                                          AIMessage(query_prompt_one_shot_output),
                                          HumanMessage(query_prompt_template.format(text))]).format_prompt()
        for text in all_queries
    ]
    print(client.llm_engine.model_config.served_model_name)
    if 'meta-llama/Llama-3' in client.llm_engine.model_config.served_model_name:
        prompts = [langchain_message_to_llama_3_prompt(ner_messages.to_messages()) for ner_messages in query_ner_prompts]
    else:
        prompts = [ner_messages.to_string() for ner_messages in query_ner_prompts]

    vllm_output = client.generate(
        prompts,
        sampling_params=vllm.SamplingParams(max_tokens=512, temperature=0),
        guided_options_request=vllm.model_executor.guided_decoding.guided_fields.GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE['ner'])
    )
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    # all_responses = [extract_json_dict(response) for response in all_responses]
    all_total_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
    return all_responses, all_total_tokens


def query_ner_parallel(dataset: str, llm: str, model_name: str, num_processes: int, num_gpus: int = 4):
    client = init_langchain_model(llm, model_name, num_gpus=num_gpus)  # LangChain model
    output_file = f'output/{dataset}_{model_name.replace("/", "_")}_queries.named_entity_output.tsv'

    queries_df = pd.read_json(f'data/{dataset}.json')

    if 'hotpotqa' in dataset:
        queries_df = queries_df[['question']]
        queries_df['0'] = queries_df['question']
        queries_df['query'] = queries_df['question']
        query_name = 'query'
    else:
        query_name = 'question'

    try:
        output_df = pd.read_csv(output_file, sep='\t')
    except:
        output_df = []

    import vllm
    if isinstance(client, vllm.LLM):
        all_queries = queries_df[query_name].values.tolist()
        all_outputs, all_num_tokens = run_ner_on_texts_vllm(client, all_queries)
        queries_df['triples'] = all_outputs
        if isinstance(all_num_tokens, list):
            all_num_tokens = sum(all_num_tokens)
        queries_df['triples'] = all_outputs
        queries_df.to_csv(output_file, sep='\t')
        print('Passage NER saved to', output_file)
        print('Total tokens:', all_num_tokens)
        return
    try:

        if len(queries_df) != len(output_df):
            queries = queries_df[query_name].values

            splits = np.array_split(range(len(queries)), num_processes)

            args = []

            for split in splits:
                args.append([queries[i] for i in split])

            if num_processes == 1:
                outputs = [run_ner_on_texts(client, args[0])]
            else:
                with ThreadPoolExecutor(max_workers=num_processes) as executor:
                    outputs = list(executor.map(partial(run_ner_on_texts, client), args))
                # with Pool(processes=num_processes) as pool:
                #     outputs = pool.map(partial_func, args)

            chatgpt_total_tokens = 0

            query_triples = []

            for output in outputs:
                query_triples.extend(output[0])
                chatgpt_total_tokens += output[1]

            queries_df['triples'] = query_triples
            queries_df.to_csv(output_file, sep='\t')
            print('Passage NER saved to', output_file)
        else:
            print('Passage NER already saved to', output_file)
    except Exception as e:
        print('No queries will be processed for later retrieval.', e)


if __name__ == '__main__':
    # Get the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=10, help='Number of processes')

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name

    query_ner_parallel(args.dataset, args.llm, args.model_name, args.num_processes)
