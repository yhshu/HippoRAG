from typing import Union

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from src.hipporag import get_query_instruction, HippoRAG
from src.langchain_util import init_langchain_model


def generate_nodes(client):
    messages = [
        SystemMessage(),
        HumanMessage(),
        AIMessage(),
        HumanMessage(),
    ]
    messages = ChatPromptTemplate.from_messages(messages).format_prompt()
    completion = client.invoke(messages.to_messages(), temperature=0, response_format={"type": "json_object"})


def link_by_passage_node(hipporag: HippoRAG, query: str, link_top_k: Union[None, int]):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.docs_to_phrases_mat, query_embedding.T)  # (num_docs, dim) x (1, dim).T = (num_docs, 1)
    query_doc_scores = np.squeeze(query_doc_scores)

    top_doc_idx = np.argsort(query_doc_scores)[-10:][::-1].tolist()
    for doc_idx in top_doc_idx:
        phrases = hipporag.get_phrase_in_doc_by_idx(doc_idx)
        pass # todo

    generate_nodes(hipporag.client)
