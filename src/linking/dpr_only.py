import json

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.hipporag import get_query_instruction, HippoRAG

rerank_system_prompt = """Given a query and the top-k retrieved passages from two retrieval methods, rerank the passages based on their relevance to the query.

- Relevance is defined as supporting or contradicting the query, with a preference for supporting facts in the case of questions.
- The first 10 passages come from a dense retrieval method, offering generally higher quality but potentially lacking in multi-hop reasoning.
- The remaining passages are from a graph-based retriever, which may be overlooked by dense retrieval but better captures connections between concepts.
- Return a JSON list of passage IDs, ordered by relevance, e.g., {"passage_id": [1, 3, 2]}."""


def dense_passage_retrieval(hipporag: HippoRAG, query: str, rerank: bool = False, logs=None):
    if 'colbertv2' in hipporag.linking_retriever_name:
        from colbert.data import Queries
        queries = Queries(path=None, data={0: query})
        query_doc_scores = np.zeros(len(hipporag.dataset_df))
        ranking = hipporag.corpus_searcher.search_all(queries, k=len(hipporag.dataset_df))
        for corpus_id, rank, score in ranking.data[0]:
            query_doc_scores[corpus_id] = score
    else:  # HuggingFace dense retrieval
        query_embedding = hipporag.embed_model.encode_text(query,
                                                           instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
                                                           return_cpu=True, return_numpy=True, norm=True)
        query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)
        query_doc_scores = query_doc_scores.T[0]

    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_scores = query_doc_scores[sorted_doc_ids.tolist()]
    logs = {} if logs is None else logs

    if rerank:
        docs = []
        for doc_id in sorted_doc_ids[:20]:
            docs.append(hipporag.dataset_df.iloc[doc_id]['paragraph'])

        user_prompt = f"Query: {query}"
        for idx, doc in enumerate(docs):
            user_prompt += f"\n\nPassage {idx + 1}:\n{doc}"

        messages = [SystemMessage(rerank_system_prompt),
                    HumanMessage(user_prompt)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        completion = hipporag.client.invoke(messages.to_messages(), temperature=0, response_format={"type": "json_object"})
        contents = completion.content

        try:
            response = json.loads(contents)
            sorted_doc_ids = [sorted_doc_ids[i - 1] for i in response['passage_id']]
            sorted_scores = [sorted_scores[i - 1] for i in response['passage_id']]
            sorted_doc_ids = np.array(sorted_doc_ids)
            sorted_scores = np.array(sorted_scores)
        except Exception as e:
            hipporag.logger.exception(f"Error parsing the response: {e}")
            return sorted_doc_ids, sorted_scores, logs

    return sorted_doc_ids, sorted_scores, logs
