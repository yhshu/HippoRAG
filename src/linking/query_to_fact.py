import numpy as np

from src.hipporag import HippoRAG, get_query_instruction_for_tasks
from src.linking import graph_search_with_entities


def link_query_to_fact_core(hipporag: HippoRAG, query, candidate_triples: list, fact_embeddings, link_top_k, graph_search=True, rerank_model_name=None):
    query_doc_scores = np.zeros(hipporag.docs_to_phrases_mat.shape[0])
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction_for_tasks(hipporag.embed_model, 'query_to_fact'),
                                                       return_cpu=True, return_numpy=True, norm=True)
    # rank and get link_top_k oracle facts given the query
    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)  # (num_facts, dim) x (1, dim).T = (num_facts, 1)
    query_fact_scores = np.squeeze(query_fact_scores)
    if rerank_model_name is not None:
        from src.rerank import LLMLogitsReranker
        reranker = LLMLogitsReranker(rerank_model_name)
        # from src.rerank import RankGPT
        # reranker = RankGPT(rerank_model_name)
        candidate_fact_indices = np.argsort(query_fact_scores)[-30:][::-1].tolist()
        candidate_facts = [candidate_triples[i] for i in candidate_fact_indices]
        top_k_fact_indicies, top_k_facts = reranker.rerank('query_to_fact', query, candidate_fact_indices, candidate_facts, link_top_k)
        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
    else:
        if link_top_k is not None:
            top_k_fact_indicies = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        else:
            top_k_fact_indicies = np.argsort(query_fact_scores)[::-1].tolist()
        top_k_facts = [candidate_triples[i] for i in top_k_fact_indicies]

    for rank, f in enumerate(top_k_facts):
        try:
            triple_tuple = tuple([phrase.lower() for phrase in f])
            retrieved_fact_id = hipporag.triplet_to_id_dict.get(triple_tuple)
        except Exception as e:
            hipporag.logger.exception(f'Fact not found in the graph: {f}, {e}')
            continue
        else:
            fact_score = query_fact_scores[top_k_fact_indicies[rank]]
            related_doc_ids = hipporag.triple_to_docs.get(retrieved_fact_id, [])
            for doc_id in related_doc_ids:
                query_doc_scores[doc_id] += fact_score

    if not graph_search:  # only fact linking, no graph search
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_scores = query_doc_scores[sorted_doc_ids.tolist()]
        logs = None
    else:  # graph search
        # from retrieved fact to nodes in the fact
        sorted_doc_ids, sorted_scores, logs = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, query_fact_scores, top_k_facts, top_k_fact_indicies)

    if rerank_model_name is not None:
        logs['rerank'] = rerank_log
    return sorted_doc_ids, sorted_scores, logs


def graph_search_with_fact_entities(hipporag: HippoRAG, link_top_k, query_doc_scores, query_fact_scores, top_k_facts, top_k_fact_indices):
    all_phrase_weights = np.zeros(len(hipporag.node_phrases))
    linking_score_map = {}
    for rank, f in enumerate(top_k_facts):
        subject_phrase = f[0].lower()
        predicate_phrase = f[1].lower()
        object_phrase = f[2].lower()
        fact_score = query_fact_scores[top_k_fact_indices[rank]]
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
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
    assert sum(all_phrase_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'
    sorted_doc_ids, sorted_scores, logs = graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map, query_doc_scores=query_doc_scores)
    return sorted_doc_ids, sorted_scores, logs


def oracle_query_to_fact(hipporag: HippoRAG, query: str, oracle_triples: list, link_top_k, graph_search=True):
    # using query to score facts and using facts to score documents
    oracle_triples_str = [str(t) for t in oracle_triples]
    fact_embeddings = hipporag.embed_model.encode_text(oracle_triples_str, return_cpu=True, return_numpy=True, norm=True)
    return link_query_to_fact_core(hipporag, query, oracle_triples, fact_embeddings, link_top_k, graph_search)


def link_fact_by_dpr(hipporag: HippoRAG, query: str, link_top_k=10, graph_search=True):
    """
    Retrieve the most similar facts given the query
    @param hipporag: HippoRAG object
    @param query: query text
    @param link_top_k:
    @return: sorted_doc_ids (np.ndarray), sorted_scores (np.ndarray)
    """
    return link_query_to_fact_core(hipporag, query, hipporag.triples, hipporag.triple_embeddings, link_top_k, graph_search)
