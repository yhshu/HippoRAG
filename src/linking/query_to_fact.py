import numpy as np

from src.hipporag import HippoRAG
from src.linking.query_to_node import graph_search_with_entities


def oracle_query_to_fact(hipporag, query, oracle_triples, link_top_k):
    query_doc_scores = np.zeros(hipporag.docs_to_phrases_mat.shape[0])
    # using query to score facts and using facts to score documents
    query_embedding = hipporag.embed_model.encode_text(query, return_cpu=True, return_numpy=True, norm=True)
    oracle_triples_str = [str(t) for t in oracle_triples]
    fact_embeddings = hipporag.embed_model.encode_text(oracle_triples_str, return_cpu=True, return_numpy=True, norm=True)
    # rank and get link_top_k oracle facts given the query
    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)  # (num_facts, dim) x (1, dim).T = (num_facts, 1)
    top_k_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()[0]
    top_k_facts = [oracle_triples[i] for i in top_k_indices]
    for rank, f in enumerate(top_k_facts):
        try:
            triple_tuple = tuple([phrase.lower() for phrase in f])
            retrieved_fact_id = hipporag.triplet_fact_to_id_dict.get(triple_tuple)
        except Exception as e:
            hipporag.logger.exception(f'Fact not found in the graph: {f}, {e}')
            continue
        else:
            fact_score = query_fact_scores[top_k_indices[rank]][0]
            for doc_id_fact_id in hipporag.docs_to_facts:
                corpus_id = doc_id_fact_id[0]
                fact_id = doc_id_fact_id[1]
                if fact_id == retrieved_fact_id:
                    query_doc_scores[corpus_id] += fact_score
    sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
    sorted_scores = query_doc_scores[sorted_doc_ids.tolist()]
    return sorted_doc_ids, sorted_scores


def link_fact_by_dpr(hipporag: HippoRAG, query: str, top_k=10, graph_algorithm=False):
    """
    Retrieve the most similar facts given the query
    @param hipporag: HippoRAG object
    @param query: query text
    @param top_k:
    @return: sorted_doc_ids (np.ndarray), sorted_scores (np.ndarray)
    """
    query_doc_scores = np.zeros(hipporag.docs_to_phrases_mat.shape[0])
    # using query to score facts and using facts to score documents
    query_embedding = hipporag.embed_model.encode_text(query, return_cpu=True, return_numpy=True, norm=True)
    # rank and get link_top_k oracle facts given the query
    query_fact_scores = np.dot(hipporag.fact_embeddings, query_embedding.T)  # (num_facts, dim) x (1, dim).T = (num_facts, 1)

    top_k_indices = np.argsort(query_fact_scores)[-top_k:][::-1].tolist()[0]
    top_k_facts = [hipporag.triplet_facts[i] for i in top_k_indices]

    for rank, f in enumerate(top_k_facts):
        try:
            triple_tuple = tuple([phrase.lower() for phrase in f])
            retrieved_fact_id = hipporag.triplet_fact_to_id_dict.get(triple_tuple)
        except Exception as e:
            hipporag.logger.exception(f'Fact not found in the graph: {f}, {e}')
            continue
        else:
            fact_score = query_fact_scores[top_k_indices[rank]][0]
            for doc_id_fact_id in hipporag.docs_to_facts:
                corpus_id = doc_id_fact_id[0]
                fact_id = doc_id_fact_id[1]
                if fact_id == retrieved_fact_id:
                    query_doc_scores[corpus_id] += fact_score

    if not graph_algorithm:  # only fact linking, no graph search
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_scores = query_doc_scores[sorted_doc_ids.tolist()]
    else:  # graph search
        # from retrieved fact to nodes in the fact
        all_phrase_weights = np.zeros(len(hipporag.node_phrases))
        phrase_score = {}
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[top_k_indices[rank]][0]
            for phrase in [subject_phrase, object_phrase]:
                phrase_id = hipporag.kb_node_phrase_to_id.get(phrase, None)
                if phrase_id:
                    all_phrase_weights[phrase_id] = 1.0
                    if hipporag.node_specificity and hipporag.phrase_to_num_doc[phrase_id] != 0:
                        all_phrase_weights[phrase_id] = 1 / hipporag.phrase_to_num_doc[phrase_id]
            phrase_score[subject_phrase] = fact_score
            phrase_score[object_phrase] = fact_score

        linking_score_map = {hipporag.node_phrases[phrase_id]: score for phrase_id, score in phrase_score.items()}
        logs, sorted_doc_ids, sorted_scores = graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map, query_doc_scores=None)
    return sorted_doc_ids, sorted_scores
