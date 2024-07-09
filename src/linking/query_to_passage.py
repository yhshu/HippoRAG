from typing import Union

import numpy as np

from src.hipporag import HippoRAG, get_query_instruction_for_datasets


def linking_by_passage(hipporag: HippoRAG,query: str, link_top_k: Union[None, int]):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction_for_datasets(
        hipporag.embed_model, hipporag.corpus_name),
                                                   return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)
    query_doc_scores = query_doc_scores.T[0]

    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

    facts_by_doc = []
    for doc_id, doc_score in zip(sorted_doc_ids[:10], sorted_doc_scores[:10]):
        facts = hipporag.docs_to_facts