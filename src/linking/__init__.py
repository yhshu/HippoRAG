import numpy as np

from src.processing import min_max_normalize


def graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map, query_doc_scores=None):
    """
    Run Personalized PageRank (PPR) or other graph algorithm to get doc rankings
    @param query_doc_scores: query-doc scores based on dense model, without phrase as mediators
    @return:
    """

    if not hipporag.dpr_only:
        combined_vector = np.max([all_phrase_weights], axis=0)

        if hipporag.graph_alg == 'ppr':
            ppr_phrase_probs = hipporag.run_pagerank_igraph_chunk([all_phrase_weights])[0]
        elif hipporag.graph_alg == 'none':
            ppr_phrase_probs = combined_vector
        elif hipporag.graph_alg == 'neighbor_2':
            ppr_phrase_probs = hipporag.get_neighbors(combined_vector, 2)
        elif hipporag.graph_alg == 'neighbor_3':
            ppr_phrase_probs = hipporag.get_neighbors(combined_vector, 3)
        elif hipporag.graph_alg == 'paths':
            ppr_phrase_probs = hipporag.get_neighbors(combined_vector, 3)
        else:
            assert False, f'Graph Algorithm {hipporag.graph_alg} Not Implemented'

        fact_prob = hipporag.triples_to_phrases_mat.dot(ppr_phrase_probs)
        ppr_doc_prob = hipporag.docs_to_triples_mat.dot(fact_prob)
        ppr_doc_prob = min_max_normalize(ppr_doc_prob)
    # Combine Query-Doc and PPR Scores
    if hipporag.doc_ensemble or hipporag.dpr_only:
        # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
        if np.min(list(linking_score_map.values())) > hipporag.recognition_threshold:  # high confidence in named entities
            doc_prob = ppr_doc_prob
            hipporag.statistics['ppr'] = hipporag.statistics.get('ppr', 0) + 1
        else:  # relatively low confidence in named entities, combine the two scores
            # the higher threshold, the higher chance to use the doc ensemble
            doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
            query_doc_scores = min_max_normalize(query_doc_scores)

            top_ppr = np.argsort(ppr_doc_prob)[::-1][:10]
            top_ppr = [(top, ppr_doc_prob[top]) for top in top_ppr]

            top_doc = np.argsort(query_doc_scores)[::-1][:10]
            top_doc = [(top, query_doc_scores[top]) for top in top_doc]

            top_hybrid = np.argsort(doc_prob)[::-1][:10]
            top_hybrid = [(top, doc_prob[top]) for top in top_hybrid]

            hipporag.ensembling_debug.append((top_ppr, top_doc, top_hybrid))
            hipporag.statistics['ppr_doc_ensemble'] = hipporag.statistics.get('ppr_doc_ensemble', 0) + 1
    else:
        doc_prob = ppr_doc_prob
    # Return ranked docs and ranked scores
    sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
    sorted_scores = doc_prob[sorted_doc_ids]
    if not hipporag.dpr_only:
        # logs
        phrase_one_hop_triples = []
        for phrase_id in np.where(all_phrase_weights > 0.001)[0]:
            # get all the triples that contain the phrase from self.graph_plus
            for t in list(hipporag.kg_adj_list[phrase_id].items())[:20]:
                phrase_one_hop_triples.append([hipporag.node_phrases[t[0]], t[1]])
            for t in list(hipporag.kg_inverse_adj_list[phrase_id].items())[:20]:
                phrase_one_hop_triples.append([hipporag.node_phrases[t[0]], t[1], 'inv'])

        # get top ranked nodes from doc_prob and self.doc_to_phrases_mat
        nodes_in_retrieved_doc = []
        for doc_id in sorted_doc_ids[:5]:
            node_id_in_doc = list(np.where(hipporag.docs_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
            nodes_in_retrieved_doc.append([hipporag.node_phrases[node_id] for node_id in node_id_in_doc])

        # get top ppr_phrase_probs
        top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]

        # get phrases for top_pagerank_phrase_ids
        top_ranked_nodes = [hipporag.node_phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]

        logs = {'linked_node_scores': {k: float(v) for k, v in linking_score_map.items()},
                '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                'top_ranked_nodes': top_ranked_nodes, 'nodes_in_retrieved_doc': nodes_in_retrieved_doc}
    else:
        logs = {}
    return sorted_doc_ids, sorted_scores, logs
