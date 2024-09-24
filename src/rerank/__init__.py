import difflib
import json
import os
import re
from typing import List, Tuple

import torch
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.langchain_util import init_langchain_model


class LLMLogitsCache:
    def __init__(self, model_name='gpt-4o-mini'):
        self.set_model_name(model_name)

    def set_model_name(self, model_name):
        self.file_path = f'.llm_logits_{model_name}.pkl'
        self.cache = self.load()

    def load(self):
        import pickle
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def get(self, prompt: str):
        return self.cache.get(prompt, None)

    def set(self, prompt: str, top_logprobs: list):
        self.cache[prompt] = top_logprobs
        self.save()

    def save(self):
        import pickle
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.cache, f)


llm_logits_cache = LLMLogitsCache()


def format_candidates(candidates: list):
    return '\n'.join([f'{chr(97 + i)}. {candidate}' for i, candidate in enumerate(candidates)])


class Reranker:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name.startswith('gpt') or model_name.startswith('ft:gpt') or model_name.startswith('o1-'):
            llm_provider = 'openai'
        elif model_name.lower().startswith('gritlm'):
            llm_provider = 'gritlm'
        elif model_name == 'llama_cpp_server':
            llm_provider = 'llama_cpp_server'
            model_name = 'http://localhost:8080/completion'
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for reranker.")
        self.model = init_langchain_model(llm_provider, model_name)

    def rerank(self, task: str, query, candidate_indices, candidate_items, top_k=None):
        pass


class LLMLogitsReranker(Reranker):

    def __init__(self, model_name):
        super().__init__(model_name)
        llm_logits_cache.set_model_name(model_name)

    def rerank(self, task, query, candidate_items, candidate_indices=None, passage='', top_k=None):
        if candidate_indices is None:
            candidate_indices = list(range(len(candidate_items)))
        if task == 'fact_reranking':
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        "Given the following query and candidate facts, choose the most relevant fact based on the query. "
                        "Please choose an option by **giving a single latter **, e.g., `a`. "
                        "Here is an example."),
                    HumanMessage(
                        "Query: 'What is the capital of France?'\n"
                        "Candidate facts: \n"
                        "a. ('France', 'capital', 'Berlin')\n"
                        "b. ('France', 'capital', 'Paris')\n"
                        "c. ('France', 'language', 'French')\nChoice: "),
                    AIMessage("b"),
                    HumanMessagePromptTemplate.from_template("Query: {query}\nCandidate facts: \n{candidate_items}\nChoice: ")
                ]
            )
            formatted_prompt = prompt_template.format_prompt(query=query, candidate_items=format_candidates(candidate_items[:26]))
        elif task == 'passage_fact_reranking':
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        "Given the following query, a passage and candidate facts extracted from this passage, choose the most relevant fact based on the query. "
                        "The passage is potentially relevant to the query, so some of these facts may be relevant to the query. "
                        "Please choose an option by **giving a single latter **, e.g., `a`. "
                        "Here is an example."),
                    HumanMessage(
                        "Query: 'What is the capital of France?'\n"
                        "Passage: 'France is a country in Europe. The capital of France is Paris.'\n"
                        "Candidate facts: \n"
                        "a. ('France', 'capital', 'Berlin')\n"
                        "b. ('France', 'capital', 'Paris')\n"
                        "c. ('France', 'language', 'French')\nChoice: "),
                    AIMessage("b"),
                    HumanMessagePromptTemplate.from_template("Query: {query}\nPassage: {passage}\nCandidate facts: \n{candidate_items}\nChoice: ")
                ]
            )
            formatted_prompt = prompt_template.format_prompt(query=query, passage=passage, candidate_items=format_candidates(candidate_items[:26]))
        else:
            raise NotImplementedError(f"Task {task} not implemented.")

        logit_bais = {token_id: 100 for token_id in range(64, 64 + 26)}  # 'a' to 'z' tokens
        logit_bais.update({6: -100, 7: -100, 8: -100, 9: -100, 12: -100, 13: -100, 220: -100, 334: -100, 4155: -100, 12488: -100})

        top_logprobs = llm_logits_cache.get(formatted_prompt.to_string())
        if top_logprobs is None:
            completion = self.model.invoke(formatted_prompt.to_messages(), max_tokens=1, seed=1, logprobs=True, top_logprobs=20, logit_bias=logit_bais)
            top_logprobs = completion.response_metadata['logprobs']['content'][0]['top_logprobs']
            llm_logits_cache.set(formatted_prompt.to_string(), top_logprobs)

        top_logprobs = top_logprobs[:len(candidate_items)]
        top_scores = {}  # {candidate_item: score}
        for top_log in top_logprobs:
            try:
                top_scores[candidate_items[ord(top_log['token'].lower().strip("-().*' ")[0]) - 97]] = top_log['logprob']
            except Exception as e:
                print('score_pairs_chat exception', e, 'TopLogprob token:', top_log['token'])
                continue

        # connect candidate_indices and candidate_items
        assert len(candidate_indices) == len(candidate_items)
        candidate_indices_and_items = list(zip(candidate_indices, candidate_items))

        # reorder candidate_indices based on top_scores
        scored_candidates = [candidate_indices_and_items[i] for i in range(0, len(candidate_indices_and_items)) if candidate_indices_and_items[i][1] in top_scores]
        sorted_scored_indices_and_items = sorted(scored_candidates, key=lambda x: top_scores[x[1]], reverse=True)

        # split candidate_indices and candidate_items
        sorted_candidate_indices, sorted_candidate_items = zip(*sorted_scored_indices_and_items)
        sorted_scores = [top_scores[item] for item in sorted_candidate_items]

        # return top_k indices and items
        return sorted_candidate_indices[:top_k], sorted_candidate_items[:top_k]  # sorted_scores[:top_k]


class RankGPT(Reranker):
    """
    https://arxiv.org/pdf/2304.09542
    """

    def __init__(self, model_name):
        super().__init__(model_name)
        set_llm_cache(SQLiteCache(database_path=f".llm_{model_name}_rerank.db"))

    def rerank(self, task: str, query, candidate_items, candidate_indices, top_k=None, window_size=4, step_size=2):
        if candidate_indices is None:
            candidate_indices = list(range(len(candidate_items)))

        def parse_numbers(s):
            try:
                numbers = re.findall(r'\[(\d+)\]', s)
                numbers = [int(num) for num in numbers]
                return numbers
            except Exception as e:
                print('parse_numbers exception', e)
                return [i for i in range(len(window_items))]

        if task == 'query_to_fact':
            assert len(candidate_indices) == len(candidate_items)

            # Initialize the result list with the original order
            result_indices = list(range(len(candidate_items)))

            # Process the passages in reverse order using sliding windows
            for start in range(len(candidate_items) - window_size, -1, -step_size):
                end = start + window_size
                window_items = candidate_items[start:end]
                window_indices = list(range(start, end))

                messages = self.write_rerank_prompt(query, window_items)

                completion = self.model.invoke(messages)
                content = completion.content
                local_indices = parse_numbers(content)

                # Update the result list based on the local ranking
                global_indices = [window_indices[i] for i in local_indices]
                for i, global_index in enumerate(global_indices):
                    if global_index in result_indices:
                        result_indices.remove(global_index)
                    result_indices.insert(i, global_index)

            # Reorder candidates based on the final ranking
            sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
            sorted_candidate_items = [candidate_items[i] for i in result_indices]

            return sorted_candidate_indices[:top_k], sorted_candidate_items[:top_k]

    def write_rerank_prompt(self, query, window_items):
        messages = [
            SystemMessage("You are RankGPT, an intelligent assistant that can rank facts based on their relevancy to the query."),
            HumanMessage(f"I will provide you with {len(window_items)} facts, each indicated by number identifier []. "
                         f"Rank them based on their relevance to query: {query}"),
            AIMessage("Okay, please provide the facts"),
        ]
        for i, item in enumerate(window_items):
            messages.append(HumanMessage(f"[{i}] {item}"))
            messages.append(AIMessage(f"Received fact [{i}]"))
        messages.append(HumanMessage(f"Search Query: {query}\nRank the {len(window_items)} facts above based on their relevance to the search query. "
                                     "The facts should be listed in descending order using identifiers, "
                                     "and the most relevant facts should be listed first, and the output format should be [] > [], e.g., [1] > [2]. "
                                     "Only respond the ranking results, do not say any word or explain."))
        return messages


generative_reranking_prompt = """You are an expert in ranking facts based on their relevance to the query. 

- Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
- If the query is a claim, relevance means the fact supports or contradicts it. For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
- Select up to 4 relevant facts from the candidate list and output in JSON format without any other words, e.g., 

```json
{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
```

- If no facts are relevant, return an empty list, e.g., {"fact": []}.
- Only use facts from the candidate list; do NOT generate new facts.
"""
generative_cot_reranking_prompt = """You are an expert in ranking facts based on their relevance to the query. 

- Multi-hop reasoning **may be** required, meaning you might need to combine multiple facts to form a complete response.
- If the query is a claim, relevance means the fact supports or contradicts it. For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
- Provide a rationale and select up to 4 relevant facts from the candidate list, output in JSON format without any other words, e.g., 

```json
{"thought": "Fact (s1, p1, o1) and (s2, p2, o2) support this query.", "fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
```

- If no facts are relevant, return an empty list, e.g., {"thought": "No fact is relevant to this query.", "fact": []}.
- Only use facts from the candidate list; do NOT generate new facts.
"""


def merge_messages(messages: List, model_name: str):
    if model_name.startswith('o1-'):
        # concatenate message contents and merge them into one HumanMessage
        message_contents = '\n'.join([message.content for message in messages])
        return [HumanMessage(message_contents)]
    return messages


class LLMGenerativeReranker(Reranker):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        if model_name.startswith('gpt-') or model_name.startswith('ft:gpt') or model_name.startswith('o1-'):
            set_llm_cache(SQLiteCache(database_path=f".llm_{model_name}_rerank.db"))

    def rerank(self, task: str, query: str, candidate_items: List[Tuple], candidate_indices, top_k=None):
        if candidate_indices is None:
            candidate_indices = list(range(len(candidate_items)))

        if task == 'fact_reranking':
            messages = self.write_rerank_prompt(query, candidate_items)
            if self.model_name.startswith('o1-'):
                completion = self.model.invoke(messages, temperature=None, response_format={"type": "json_object"})
            else:
                completion = self.model.invoke(messages, temperature=0, response_format={"type": "json_object"})
            content = completion.content
            try:
                response = json.loads(content)
            except Exception as e:
                print('json.load exception', e, 'output:', content)
                response = {'fact': []}

            result_indices = []
            for generated_fact in response['fact']:
                closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)[0]
                try:
                    result_indices.append(candidate_items.index(eval(closest_matched_fact)))
                except Exception as e:
                    print('result_indices exception', e)

            sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
            sorted_candidate_items = [candidate_items[i] for i in result_indices]
            return sorted_candidate_indices[:top_k], sorted_candidate_items[:top_k]

    def write_rerank_prompt(self, query: str, candidates: List[Tuple], group_by_subject=False):
        user_prompt = 'Query: ' + query
        # sort candidate triples by subject
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        # group by subject
        candidates_group_by_subject = []  # [[triple]]
        for i, candidate in enumerate(sorted_candidates):
            if i == 0 or candidate[0] != sorted_candidates[i - 1][0]:
                candidates_group_by_subject.append([candidate])
            else:
                candidates_group_by_subject[-1].append(candidate)

        if not group_by_subject:
            user_prompt += '\nCandidate facts:\n'
            for i, candidate in enumerate(sorted_candidates):
                user_prompt += f'- {json.dumps(list(candidate))}\n'
        else:
            user_prompt += '\nCandidate facts:\n'
            for i, candidates in enumerate(candidates_group_by_subject):  # write each subject group in one line
                user_prompt += f'- {json.dumps([list(candidate) for candidate in candidates])}\n'

        messages = [
            SystemMessage(generative_reranking_prompt),
            HumanMessage(user_prompt),
        ]
        messages = merge_messages(messages, self.model_name)
        return messages


def retrieved_to_candidate_facts(candidate_items, candidate_indices, k=30):
    # bind candidate_items and candidate_indices
    candidate_indices_and_items = list(zip(candidate_indices, candidate_items))
    # remove triples with duplicate subjects and objects
    candidate_dict = {}
    for item in candidate_indices_and_items:
        if (item[1][0], item[1][2]) not in candidate_dict:
            candidate_dict[(item[1][0], item[1][2])] = item
    candidate_indices_and_items = list(candidate_dict.values())
    # order candidate items by subject
    candidate_indices_and_items = sorted(candidate_indices_and_items, key=lambda x: x[1][0])
    # unpack candidate_indices_and_items
    sorted_candidate_indices, sorted_candidate_items = zip(*candidate_indices_and_items)
    return sorted_candidate_items[:k], sorted_candidate_indices[:k]


class HFLoRAModelGenerativeReranker(Reranker):
    def __init__(self, model_path, base_model='meta-llama/Meta-Llama-3.1-8B-Instruct'):
        if 'adapter_config.json' in os.listdir(model_path):
            base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto', return_dict=True)
            model = PeftModel.from_pretrained(base_model, model_path)
            self.model = model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto', return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def rerank(self, task: str, query: str, input_items: List[Tuple], input_indices, len_after_rerank=None):
        if task == 'fact_reranking':
            candidate_items, candidate_indices = retrieved_to_candidate_facts(input_items, input_indices, k=30)

            messages = [{'role': 'system', 'content': generative_reranking_prompt},
                        {'role': 'user', 'content': f'\nQuery: {query}\nCandidate facts:\n' + '\n'.join([json.dumps(triple).lower() for triple in candidate_items])}]

            with torch.no_grad():
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
                input_length = inputs['input_ids'].shape[-1]

                outputs = self.model.generate(**inputs, max_length=1200, pad_token_id=self.tokenizer.eos_token_id)
                completion = outputs[:, input_length:]

                output_text = self.tokenizer.decode(completion[0])
                output_text = output_text.split('<|end_header_id|>')[1].split('<|eot_id|>')[0].strip()

            try:
                response = json.loads(output_text)
            except Exception as e:
                print('json.load exception', e, 'output:', output_text)
                response = {'fact': []}

            result_indices = []
            for generated_fact in response['fact']:
                closest_matched_fact = difflib.get_close_matches(json.dumps(generated_fact), [json.dumps(i) for i in candidate_items], n=1, cutoff=0.0)[0]
                try:
                    result_indices.append(candidate_items.index(tuple(eval(closest_matched_fact))))
                except Exception as e:
                    print('result_indices exception', e)

            sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
            sorted_candidate_items = [candidate_items[i] for i in result_indices]
            return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank]


class RerankerPlaceholder(Reranker):
    def __init__(self, model_name):
        self.model_name = model_name

    def rerank(self, task: str, query, input_items, input_indices, top_k=None):
        return input_indices[:top_k], input_items[:top_k]
