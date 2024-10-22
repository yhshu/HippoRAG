import argparse
import os

import tiktoken

from src.lm_wrapper.gritlm import GritLMLangchainWrapper
from src.util.llama_cpp_service import LlamaCppWrapper

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


class LangChainModel:
    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs


def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == 'openai':
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
        from langchain_openai import ChatOpenAI
        assert model_name.startswith('gpt-') or model_name.startswith('ft:gpt-') or model_name.startswith('o1-')
        if model_name.startswith('o1-'):
            return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, max_retries=max_retries, timeout=timeout, **kwargs)
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
    elif llm == 'together':
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/
        from langchain_together import ChatTogether
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature, **kwargs)
    elif llm == 'ollama':
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name)  # e.g., 'llama3'
    elif llm == 'llama.cpp':
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/
        from langchain_community.chat_models import ChatLlamaCpp
        return ChatLlamaCpp(model_path=model_name, verbose=True)  # model_name is the model path (gguf file)
    elif llm == 'llama_cpp_server':
        return LlamaCppWrapper()
    elif llm.lower() == 'gritlm':
        return GritLMLangchainWrapper(model_name=model_name)
    elif llm == 'vllm_openai':
        from langchain_community.llms.vllm import VLLMOpenAI
        return VLLMOpenAI(openai_api_key='osunlp', openai_api_base='http://localhost:8000/v1', model_name=model_name)
    elif llm == 'vllm':
        from vllm import LLM
        tensor_parallel_size = kwargs.get('num_gpus', 4)
        llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, seed=0, dtype='auto', max_seq_len_to_capture=4096, enable_prefix_caching=True,
        enforce_eager=True,)
        return llm
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--query', type=str, help='query text', default="who are you?")
    args = parser.parse_args()

    model = init_langchain_model(args.llm, args.model_name)
    messages = [("system", "You are a helpful assistant. Please answer the question from the user."), ("human", args.query)]
    completion = model.invoke(messages)
    print(completion.content)
