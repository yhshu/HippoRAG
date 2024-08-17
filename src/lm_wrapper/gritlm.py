# See https://github.com/ContextualAI/gritlm
from typing import Union, List

import numpy as np
import torch
from gritlm import GritLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.lm_wrapper import EmbeddingModelWrapper


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


class GritLMWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = 'GritLM/GritLM-7B', **kwargs):
        """
        Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head).
        To load the 8x7B you will likely need multiple GPUs.
        @param model_name:
        @param kwargs:
        """
        self.model = GritLM(model_name, torch_dtype='auto', **kwargs)

    def encode_list(self, texts: list, instruction: str, batch_size=80):
        return self.model.encode(texts, instruction=gritlm_instruction(instruction), batch_size=batch_size)

    def encode_text(self, text: Union[str, List], instruction: str = '', norm=True, return_numpy=False, return_cpu=False):
        if isinstance(text, str):
            text = [text]
        if isinstance(text, list):
            res = self.encode_list(text, instruction)
        else:
            raise ValueError(f"Expected str or list, got {type(text)}")
        if isinstance(res, torch.Tensor):
            if return_cpu:
                res = res.cpu()
            if return_numpy:
                res = res.numpy()
        if norm:
            if isinstance(res, torch.Tensor):
                res = res.T.divide(torch.linalg.norm(res, dim=1)).T
            if isinstance(res, np.ndarray):
                res = (res.T / np.linalg.norm(res, axis=1)).T
        return res

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        """
        @param query_vec: query vector
        @param doc_vecs: doc matrix
        @return: a matrix of query-doc scores
        """
        return np.dot(doc_vecs, query_vec.T)

    def generate(self, messages: List, max_new_tokens=256, do_sample=False):
        """

        @param messages: a list, e.g., [{"role": "user", "content": "Please write me a poem."}]
        @return:
        """
        encoded = self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        encoded = encoded.to(self.model.device)
        gen = self.model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=do_sample)
        decoded = self.model.tokenizer.batch_decode(gen)
        return decoded


class GritLMLangchainWrapper:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = GritLMWrapper(model_name)

    def invoke(self, messages: List, max_new_tokens=256, do_sample=False, **kwargs):
        new_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                new_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                new_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                new_messages.append({"role": "assistant", "content": message.content})

        completions = self.model.generate(new_messages, max_new_tokens, do_sample)
        return AIMessage(content=completions, response_metadata={'model_name': self.model_name})
