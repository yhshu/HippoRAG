from typing import Union
from warnings import deprecated

import numpy as np
from nltk import sent_tokenize

from src.hipporag import HippoRAG, get_query_instruction
from src.linking.query_to_fact import graph_search_with_fact_entities


@deprecated
def linking_by_passage_sentences(hipporag: HippoRAG, query: str, link_top_k: Union[None, int], fact_context=None):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)
    query_doc_scores = query_doc_scores.T[0]

    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

    facts = []
    triple_to_doc_id = {}
    docs = []
    all_sentences_from_top_docs = []
    for doc_id, doc_score in zip(sorted_doc_ids[:10], sorted_doc_scores[:10]):
        doc = hipporag.corpus[doc_id]
        docs.append(doc)
        triples, triple_ids = hipporag.get_triples_and_triple_ids_by_corpus_idx(doc_id)
        facts.extend(triples)
        for t in triples:
            triple_to_doc_id[t] = doc_id

        doc_title = doc['title']
        doc_text = doc['text']
        sentences = sent_tokenize(doc_text)
        all_sentences_from_top_docs.extend(f"Title: {doc_title}\nText: {sent}" for sent in sentences)

    # rank sentences from top docs
    sentence_embeddings = hipporag.embed_model.encode_text(all_sentences_from_top_docs, return_cpu=True, return_numpy=True, norm=True)
    query_embedding = hipporag.embed_model.encode_text(query, get_query_instruction(hipporag.embed_model, 'query_to_sentence', hipporag.corpus_name),
                                                       norm=True, return_cpu=True, return_numpy=True)
    query_sentence_scores = np.dot(sentence_embeddings, query_embedding.T)
    query_sentence_scores = np.squeeze(query_sentence_scores)
    top_k_sentence_indices = np.argsort(query_sentence_scores)[-30:][::-1].tolist()
    top_k_sentences = [all_sentences_from_top_docs[i] for i in top_k_sentence_indices]

    max_fact_scores = np.zeros(len(facts))

    for sentence in top_k_sentences:
        fact_embeddings = hipporag.embed_model.encode_text([str(f) for f in facts], return_cpu=True, return_numpy=True, norm=True)
        sentence_embedding = hipporag.embed_model.encode_text(sentence, instruction='Given a sentence excerpted from a document, '
                                                                                    'and starting with the title of the document,'
                                                                                    ' retrieve the triplet facts most relevant to the sentence.',
                                                              return_cpu=True, return_numpy=True, norm=True)
        fact_sentence_scores = np.dot(fact_embeddings, sentence_embedding.T)
        fact_sentence_scores = np.squeeze(fact_sentence_scores)

        max_fact_scores = np.maximum(max_fact_scores, fact_sentence_scores)

    if link_top_k is not None:
        top_k_fact_indicies = np.argsort(max_fact_scores)[-link_top_k:][::-1].tolist()
    else:
        top_k_fact_indicies = np.argsort(max_fact_scores)[::-1].tolist()
    top_k_facts = [facts[i] for i in top_k_fact_indicies]

    sorted_doc_ids, sorted_doc_scores, logs = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, max_fact_scores, top_k_facts, top_k_fact_indicies)
    return sorted_doc_ids, sorted_doc_scores, logs


@deprecated
def linking_by_passage(hipporag: HippoRAG, query: str, link_top_k: Union[None, int]):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)
    query_doc_scores = query_doc_scores.T[0]

    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

    facts = []

    triple_to_doc_id = {}
    query_fact_scores = []
    docs = []
    for doc_id, doc_score in zip(sorted_doc_ids[:5], sorted_doc_scores[:5]):
        doc = hipporag.corpus[doc_id]
        docs.append(doc)
        triples, triple_ids = hipporag.get_triples_and_triple_ids_by_corpus_idx(doc_id)
        if len(triple_ids):
            # find the most relevant fact for each doc
            fact_embeddings = hipporag.embed_model.encode_text([str(f) for f in triples], return_cpu=True, return_numpy=True, norm=True)
            query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_fact', hipporag.corpus_name),
                                                               return_cpu=True, return_numpy=True, norm=True)
            scores = np.dot(fact_embeddings, query_embedding.T)
            scores = np.squeeze(scores)
            top_fact_idx = np.argmax(scores)
            facts.append(triples[top_fact_idx])
            for t in triples:
                triple_to_doc_id[t] = doc_id
            query_fact_scores.append(scores[top_fact_idx])

    query_fact_scores = np.array(query_fact_scores)

    sorted_doc_ids, sorted_doc_scores, logs = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, query_fact_scores, facts, list(range(len(facts))))
    return sorted_doc_ids, sorted_doc_scores, logs


def link_by_passage_fact(hipporag: HippoRAG, query: str, link_top_k: Union[None, int], fact_context=None):
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
        triples, triple_ids = hipporag.get_triples_and_triple_ids_by_corpus_idx(doc_id)
        facts.extend(triples)
        for t in triples:
            triple_to_doc_id[t] = doc_id

    # add contexts to candidate_facts
    if fact_context is not None:
        facts_to_encode = []
        for f in facts:
            doc_id = triple_to_doc_id[f]
            doc_title = hipporag.corpus[doc_id]['title']
            doc_text = hipporag.corpus[doc_id]['text']
            if fact_context == 'sentence':
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
                # print(f"Query: {query}\nFact: {f}\nFrom Passage: {doc_title}\nSource: {relevant_sentence}")
            elif fact_context == 'passage':
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
