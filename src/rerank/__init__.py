import os

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from src.langchain_util import init_langchain_model


class Reranker:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name.startswith('gpt'):
            llm_provider = 'openai'
            self.model = init_langchain_model(llm_provider, model_name)
            llm_logits_cache.set_model_name(model_name)

    def rerank(self, task: str, query, candidate_indices, candidate_items, top_k=None):
        pass


def format_candidates(candidates: list):
    return '\n'.join([f'{chr(97 + i)}. {candidate}' for i, candidate in enumerate(candidates)])


class LLMLogitsCache:
    def __init__(self, model_name='gpt-3.5-turbo'):
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


class LLMReranker(Reranker):

    def rerank(self, task, query, candidate_indices, candidate_items, top_k=None):
        if task == 'query_to_fact':
            query_to_fact_prompt_template = ChatPromptTemplate.from_messages(
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
            query_to_fact_prompt = query_to_fact_prompt_template.format_prompt(query=query, candidate_items=format_candidates(candidate_items[:26]))
            logit_bais = {token_id: 100 for token_id in range(64, 64 + 26)}  # 'a' to 'z' tokens
            logit_bais.update({6: -100, 7: -100, 8: -100, 9: -100, 12: -100, 13: -100, 220: -100, 334: -100, 4155: -100, 12488: -100})

            top_logprobs = llm_logits_cache.get(query_to_fact_prompt.to_string())
            if top_logprobs is None:
                chat_completion = self.model.invoke(query_to_fact_prompt.to_messages(), max_tokens=1, seed=1, logprobs=True, top_logprobs=20,
                                                    logit_bias=logit_bais)
                top_logprobs = chat_completion.response_metadata['logprobs']['content'][0]['top_logprobs']
                llm_logits_cache.set(query_to_fact_prompt.to_string(), top_logprobs)

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

            # return top_k indices and items
            return sorted_candidate_indices[:top_k], sorted_candidate_items[:top_k]

        else:
            raise NotImplementedError(f"Task {task} not implemented.")
