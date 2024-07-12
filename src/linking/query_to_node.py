import numpy as np

from src.hipporag import HippoRAG, get_query_instruction
from src.processing import softmax_with_zeros


def link_node_by_dpr(hipporag: HippoRAG, query: str, top_k=10):
    """
    Retrieve the most similar phrases given the query
    @param hipporag: hipporag object
    @param query: query text
    @param top_k: the number of top phrases to retrieve
    @return: all_phrase_weights, linking_score_map
    """
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_node', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)

    # Get Closest Entity Nodes
    prob_vectors = np.dot(query_embedding, hipporag.kb_node_phrase_embeddings.T)  # (1, dim) x (num_phrases, dim).T = (1, num_phrases)

    linked_phrase_ids = []
    linked_phrase_scores = []

    for prob_vector in prob_vectors:
        non_nan_mask = ~np.isnan(prob_vector)
        if top_k:
            indices = np.argsort(prob_vector[non_nan_mask])[-top_k:][::-1]  # indices of top-k phrases
        else:
            indices = np.argsort(prob_vector[non_nan_mask])[::-1]
        linked_phrase_ids.extend(indices)
        linked_phrase_scores.extend(prob_vector[indices])

    all_phrase_weights = np.zeros_like(prob_vectors[0])
    all_phrase_weights[linked_phrase_ids] = prob_vectors[0][linked_phrase_ids]

    if hipporag.node_specificity:
        for phrase_id in linked_phrase_ids:
            if hipporag.phrase_to_num_doc[phrase_id] == 0:  # just in case the phrase is not recorded in any documents
                weight = 1
            else:  # the more frequent the phrase, the less weight it gets
                weight = 1 / hipporag.phrase_to_num_doc[phrase_id]

            all_phrase_weights[phrase_id] = all_phrase_weights[phrase_id] * weight

    nan_count = np.sum(np.isnan(all_phrase_weights))
    if nan_count:
        all_phrase_weights = np.nan_to_num(all_phrase_weights, nan=0)
        hipporag.logger.info(f'Found {nan_count} NaNs in all_phrase_weights, replaced with 0s')
    all_phrase_weights = softmax_with_zeros(all_phrase_weights)

    linking_score_map = {hipporag.node_phrases[linked_phrase_id]: max_score
                         for linked_phrase_id, max_score in zip(linked_phrase_ids, linked_phrase_scores)}
    return all_phrase_weights, linking_score_map


