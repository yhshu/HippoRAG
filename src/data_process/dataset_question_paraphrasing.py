import sys

sys.path.append('.')

import argparse
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from src.langchain_util import init_langchain_model


def paraphrasing_question(question: str, client):
    messages = [
        SystemMessage("Please paraphrase the following questions from a multi-hop QA dataset.\n"
                      "- Maintain the original intention of the question.\n"
                      "- Preserve entity names and don't add extra quotation marks unless they are present in the original text.\n"
                      "- Try to change the expression as much as possible to distance it from the original text, in order to avoid data leakage in the QA task.\n"
                      "- Provide the response in JSON format."),
        HumanMessage("Question: The Move Le Body song's band is named after who?"),
        AIMessage("{\"question\": \"After whom is the band named that performs the song Move Le Body?\"}"),
        HumanMessage("Question: " + question),
    ]
    messages = ChatPromptTemplate.from_messages(messages).format_prompt()
    completion = client.invoke(messages.to_messages(), temperature=0.5, response_format={"type": "json_object"})

    try:
        new_question = json.loads(completion.content).get('question', question)
    except Exception as e:
        new_question = question
    return new_question


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai')
    parser.add_argument('--model', type=str, default='gpt-4o')
    args = parser.parse_args()

    client = init_langchain_model(args.llm, args.model)
    dataset = json.load(open(f'data/{args.dataset}.json', 'r'))
    for sample in tqdm(dataset):
        question = sample['question']
        new_question = paraphrasing_question(question, client)
        print(f"Original Question: {question}")
        print(f"Paraphrased Question: {new_question}")
        print()
        sample['question'] = new_question
        sample['original_question'] = question

    dataset_output_path = f"data/{args.dataset}_question_para.json"
    json.dump(dataset, open(dataset_output_path, 'w'), indent=4)
