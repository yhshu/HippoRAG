from typing import Union

import numpy as np
from nltk import sent_tokenize

from src.hipporag import HippoRAG, get_query_instruction
from src.linking.query_to_fact import graph_search_with_fact_entities


def linking_by_passage(hipporag: HippoRAG, query: str, link_top_k: Union[None, int], context=None):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
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
        for t in triples:
            triple_to_doc_id[t] = doc_id

    # add contexts to candidate_facts
    if context is not None:
        facts_to_encode = []
        for f in facts:
            doc_id = triple_to_doc_id[f]
            doc_title = hipporag.corpus[doc_id]['title']
            doc_text = hipporag.corpus[doc_id]['text']
            if context == 'sentence':
                sentences = sent_tokenize(doc_text)
                sentence_embeddings = hipporag.embed_model.encode_text(sentences, return_cpu=True, return_numpy=True, norm=True)
                fact_embedding = hipporag.embed_model.encode_text(str(f), instruction='Given a triplet fact, retrieve its most relevant sentence in a passage.',
                                                                  return_cpu=True, return_numpy=True, norm=True)
                fact_sentence_scores = np.dot(sentence_embeddings, fact_embedding.T)
                fact_sentence_scores = np.squeeze(fact_sentence_scores)
                # get the top-1 sentence
                top_sentence_idx = np.argmax(fact_sentence_scores)
                relevant_sentence = sentences[top_sentence_idx]
                facts_to_encode.append(f"Fact: {f}\nFrom Passage: {doc_title}\nSource: {relevant_sentence}")
            elif context == 'passage':
                facts_to_encode.append(f"Fact: {f}\nFrom Passage: {doc_title}\nSource: {doc_text}")
    else:
        facts_to_encode = [str(f) for f in facts]

    fact_embeddings = hipporag.embed_model.encode_text(facts_to_encode, return_cpu=True, return_numpy=True, norm=True)
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_fact', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)

    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)
    query_fact_scores = np.squeeze(query_fact_scores)

    if link_top_k is not None:
        top_k_fact_indicies = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
    else:
        top_k_fact_indicies = np.argsort(query_fact_scores)[::-1].tolist()
    top_k_facts = [facts[i] for i in top_k_fact_indicies]

    sorted_doc_ids, sorted_doc_scores, logs = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, query_fact_scores, top_k_facts, top_k_fact_indicies)
    return sorted_doc_ids, sorted_doc_scores, logs
