from typing import List

import requests
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


class LlamaCppWrapper:

    def __init__(self, url="http://localhost:8080/completion"):
        """
        /.../llama.cpp/llama-server -m ./models/llama-3.1-8b-instruct-f16.gguf --port 8080 --n-gpu-layers 100 --threads 1 --all-logits
        """
        self.url = url

    def invoke(self, messages: List[BaseMessage], max_tokens: int = 256, seed: int = -1, logprobs: bool = False, top_logprobs: int = None,
               temperature: float = 0.0, logit_bias=None, response_format=None):
        prompt = langchain_message_to_llama_3_prompt(messages)
        response_json = request_llama_cpp_server(prompt, self.url, n_probs=top_logprobs, seed=seed, temperature=temperature, n_predict=max_tokens)
        if logprobs is True:
            logprobs_list = get_llama_cpp_logprobs(response_json)
            logprobs_dict = {'content': [{'token': logprobs_list[0]['token'], 'logprob': logprobs_list[0]['logprob'], 'top_logprobs': logprobs_list}]}
            ai_message = AIMessage(content=response_json["content"], response_metadata={'model_name': '', 'logprobs': logprobs_dict})
        else:
            ai_message = AIMessage(content=response_json["content"], response_metadata={'model_name': response_json['generation_settings']['model']})
        return ai_message


def langchain_message_to_llama_3_prompt(messages: list):
    prompt = "<|begin_of_text|>"
    for message in messages:
        if isinstance(message, SystemMessage):
            prompt += f"<|start_header_id|>system<|end_header_id|>" + message.content + "<|eot_id|>"
        elif isinstance(message, HumanMessage):
            prompt += f"<|start_header_id|>user<|end_header_id|>" + message.content + "<|eot_id|>"
        elif isinstance(message, AIMessage):
            prompt += f"<|start_header_id|>assistant<|end_header_id|>" + message.content + "<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>"
    return prompt


def request_llama_cpp_server(prompt, url="http://localhost:8080/completion", n_probs=20, seed=1, n_predict=256, temperature=0.0):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "n_probs": n_probs,
        "seed": seed
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def get_llama_cpp_logprobs(response: dict):
    probs = response["completion_probabilities"][0]['probs']
    res = []
    for p in probs:
        res.append({'token': p['tok_str'], 'logprob': p['prob']})
    return res


if __name__ == "__main__":
    client = LlamaCppWrapper()
    response = client.invoke([HumanMessage("Who are you?")])
    print(response)
