from typing import Union

import numpy as np
from nltk import sent_tokenize

from src.hipporag import HippoRAG, get_query_instruction_for_datasets, get_query_instruction_for_tasks
from src.linking.query_to_fact import graph_search_with_fact_entities


def linking_by_passage(hipporag: HippoRAG, query: str, link_top_k: Union[None, int], rerank_model_name=None):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction_for_datasets(hipporag.embed_model, hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)
    query_doc_scores = query_doc_scores.T[0]

    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

    facts = []
    triple_to_doc_id = {}
    docs = []
    for doc_id, doc_score in zip(sorted_doc_ids[:10], sorted_doc_scores[:10]):
        doc = hipporag.corpus[doc_id]
        docs.append(doc)
        triples, triple_ids = hipporag.get_triples_by_corpus_idx(doc_id)
        facts.extend(triples)
        for t  in triples:
            triple_to_doc_id[t] = doc_id


    fact_embeddings = hipporag.embed_model.encode_text([str(item) for item in facts], return_cpu=True, return_numpy=True, norm=True)
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction_for_tasks(hipporag.embed_model, 'query_to_fact'),
                                                       return_cpu=True, return_numpy=True, norm=True)

    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)
    query_fact_scores = np.squeeze(query_fact_scores)
    if rerank_model_name is not None:
        # from src.rerank import LLMLogitsReranker
        # reranker = LLMLogitsReranker(rerank_model_name)
        from src.rerank import RankGPT
        reranker = RankGPT(rerank_model_name)
        candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
        candidate_facts = [facts[i] for i in candidate_fact_indices]

        # add contexts to candidate_facts
        candidate_facts_with_contexts = []
        for f in candidate_facts:
            doc_id = triple_to_doc_id[f]
            doc_title = hipporag.corpus[doc_id]['title']
            sentences = sent_tokenize( hipporag.corpus[doc_id]['text'])
            sentence_embeddings = hipporag.embed_model.encode_text(sentences, return_cpu=True, return_numpy=True, norm=True)
            fact_embedding = hipporag.embed_model.encode_text(str(f), instruction='Given a triplet fact, retrieve its most relevant sentence in a passage.',
                                                              return_cpu=True, return_numpy=True, norm=True)
            fact_sentence_scores = np.dot(sentence_embeddings, fact_embedding.T)
            fact_sentence_scores = np.squeeze(fact_sentence_scores)
            # get the top-1 sentence
            top_sentence_idx = np.argmax(fact_sentence_scores)
            relevant_sentence = sentences[top_sentence_idx]
            candidate_facts_with_contexts.append(f"Fact: {f}\nFrom Passage: {doc_title}\nSource: {relevant_sentence}")

        # rerank
        top_k_fact_indicies, top_k_facts = reranker.rerank('query_to_fact', query, candidate_fact_indices, candidate_facts_with_contexts, link_top_k)

        # remove contexts from top_k_facts
        top_k_facts = [facts[i] for i in top_k_fact_indicies]

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
    else:
        if link_top_k is not None:
            top_k_fact_indicies = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        else:
            top_k_fact_indicies = np.argsort(query_fact_scores)[::-1].tolist()
        top_k_facts = [facts[i] for i in top_k_fact_indicies]

    sorted_doc_ids, sorted_doc_scores, logs = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, query_fact_scores, top_k_facts, top_k_fact_indicies)
    if rerank_model_name is not None:
        logs['rerank'] = rerank_log
    return sorted_doc_ids, sorted_doc_scores, logs
