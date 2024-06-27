import numpy as np

from src.hipporag import HippoRAG
from src.linking.query_to_node import graph_search_with_entities


def link_query_to_fact(hipporag, query, candidate_triples: list, fact_embeddings, link_top_k, graph_search=True):
    query_doc_scores = np.zeros(hipporag.docs_to_phrases_mat.shape[0])
    query_embedding = hipporag.embed_model.encode_text(query, return_cpu=True, return_numpy=True, norm=True)
    # rank and get link_top_k oracle facts given the query
    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)  # (num_facts, dim) x (1, dim).T = (num_facts, 1)
    query_fact_scores = np.squeeze(query_fact_scores)
    top_k_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
    top_k_facts = [candidate_triples[i] for i in top_k_indices]

    for rank, f in enumerate(top_k_facts):
        try:
            triple_tuple = tuple([phrase.lower() for phrase in f])
            retrieved_fact_id = hipporag.triplet_fact_to_id_dict.get(triple_tuple)
        except Exception as e:
            hipporag.logger.exception(f'Fact not found in the graph: {f}, {e}')
            continue
        else:
            fact_score = query_fact_scores[top_k_indices[rank]]
            for doc_id_fact_id in hipporag.docs_to_facts:
                corpus_id = doc_id_fact_id[0]
                fact_id = doc_id_fact_id[1]
                if fact_id == retrieved_fact_id:
                    query_doc_scores[corpus_id] += fact_score

    if not graph_search:  # only fact linking, no graph search
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_scores = query_doc_scores[sorted_doc_ids.tolist()]
    else:  # graph search
        # from retrieved fact to nodes in the fact
        all_phrase_weights = np.zeros(len(hipporag.node_phrases))
        linking_score_map = {}
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[top_k_indices[rank]]
            for phrase in [subject_phrase, object_phrase]:
                phrase_id = hipporag.kb_node_phrase_to_id.get(phrase, None)
                if phrase_id:
                    all_phrase_weights[phrase_id] = 1.0
                    if hipporag.node_specificity and hipporag.phrase_to_num_doc[phrase_id] != 0:
                        all_phrase_weights[phrase_id] = 1 / hipporag.phrase_to_num_doc[phrase_id]
            linking_score_map[subject_phrase] = fact_score
            linking_score_map[object_phrase] = fact_score

        if link_top_k:
            # choose top ranked node in linking_score_map
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True))
        logs, sorted_doc_ids, sorted_scores = graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map, query_doc_scores=None)
    return sorted_doc_ids, sorted_scores


def oracle_query_to_fact(hipporag, query, oracle_triples: list, link_top_k, graph_search=True):
    # using query to score facts and using facts to score documents
    oracle_triples_str = [str(t) for t in oracle_triples]
    fact_embeddings = hipporag.embed_model.encode_text(oracle_triples_str, return_cpu=True, return_numpy=True, norm=True)
    return link_query_to_fact(hipporag, query, oracle_triples, fact_embeddings, link_top_k, graph_search)


def link_fact_by_dpr(hipporag: HippoRAG, query: str, link_top_k=10, graph_search=True):
    """
    Retrieve the most similar facts given the query
    @param hipporag: HippoRAG object
    @param query: query text
    @param link_top_k:
    @return: sorted_doc_ids (np.ndarray), sorted_scores (np.ndarray)
    """
    return link_query_to_fact(hipporag, query, hipporag.triplet_facts, hipporag.fact_embeddings, link_top_k, graph_search)
