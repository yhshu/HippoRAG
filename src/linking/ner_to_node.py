import numpy as np
from colbert.data import Queries

from src.hipporag import HippoRAG
from src.processing import min_max_normalize


def link_node_by_colbertv2(hipporag: HippoRAG, query_ner_list, link_top_k=None):
    phrase_ids = []
    max_scores = []

    for query in query_ner_list:
        queries = Queries(path=None, data={0: query})

        queries_ = [query]
        encoded_query = hipporag.phrase_searcher.encode(queries_, full_length_search=False)

        max_score = hipporag.get_colbert_max_score(query)

        ranking = hipporag.phrase_searcher.search_all(queries, k=1)
        for phrase_id, rank, score in ranking.data[0]:
            phrase = hipporag.node_phrases[phrase_id]
            phrases_ = [phrase]
            encoded_doc = hipporag.phrase_searcher.checkpoint.docFromText(phrases_).float()
            real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

            phrase_ids.append(phrase_id)
            max_scores.append(real_score / max_score)

    # choose link_top_k based on max_scores and get the corresponding phrase_ids
    if link_top_k and isinstance(link_top_k, int):
        top_k = np.argsort(max_scores)[::-1][:link_top_k]
        phrase_ids = [phrase_ids[i] for i in top_k]
        max_scores = [max_scores[i] for i in top_k]

    # create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
    top_phrase_vec = np.zeros(len(hipporag.node_phrases))

    for phrase_id in phrase_ids:
        if hipporag.node_specificity:
            if hipporag.phrase_to_num_doc[phrase_id] == 0:
                weight = 1
            else:
                weight = 1 / hipporag.phrase_to_num_doc[phrase_id]
            top_phrase_vec[phrase_id] = weight
        else:
            top_phrase_vec[phrase_id] = 1.0

    return top_phrase_vec, {(query, hipporag.node_phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, query_ner_list)}


def link_node_by_dpr(hipporag: HippoRAG, query_ner_list: list, link_top_k=None):
    """
    Retrieve the most similar phrases (as vector) in the KG given the named entities
    :param query_ner_list:
    :return:
    """
    query_ner_embeddings = hipporag.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)
    # Get Closest Entity Nodes
    prob_vectors = np.dot(query_ner_embeddings, hipporag.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)
    all_phrase_weights, linking_score_map = link_ner_to_node(hipporag, link_top_k, hipporag.node_phrases, prob_vectors, query_ner_list)
    return all_phrase_weights, linking_score_map


def link_ner_to_node(hipporag, link_top_k, candidate_phrases: list, prob_vectors, query_ner_list):
    linked_phrases = []
    max_scores = []  # max score for each named entity
    for prob_vector in prob_vectors:
        mask = np.isnan(prob_vector)
        # phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity
        linked_phrase = np.argmax(np.ma.masked_array(prob_vector, mask))
        linked_phrases.append(candidate_phrases[linked_phrase])
        max_scores.append(prob_vector[linked_phrase])

    # choose link_top_k based on max_scores and get the corresponding linked_phrase_ids
    if link_top_k and isinstance(link_top_k, int):
        top_k = np.argsort(max_scores)[::-1][:link_top_k]
        linked_phrases = [linked_phrases[i] for i in top_k]
        max_scores = [max_scores[i] for i in top_k]

    # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
    # if node_specificity is True, it's not one-hot but a weight
    all_phrase_weights = np.zeros(len(hipporag.node_phrases))
    for linked_phrase in linked_phrases:
        phrase_id = hipporag.kb_node_phrase_to_id.get(linked_phrase, None)
        if phrase_id is None:
            hipporag.logger.error(f'Phrase {linked_phrase} not found in the KG')
            continue
        if hipporag.node_specificity:
            if hipporag.phrase_to_num_doc[phrase_id] == 0:  # just in case the phrase is not recorded in any documents
                weight = 1
            else:  # the more frequent the phrase, the less weight it gets
                weight = 1 / hipporag.phrase_to_num_doc[phrase_id]

            all_phrase_weights[phrase_id] = weight
        else:
            all_phrase_weights[phrase_id] = 1.0

    linking_score_map = {(query_phrase, linked_phrase): max_score
                         for linked_phrase, max_score, query_phrase in zip(linked_phrases, max_scores, query_ner_list)}
    return all_phrase_weights, linking_score_map


def oracle_ner_to_node(hipporag: HippoRAG, query_ner_list, oracle_phrases, link_top_k=None):
    query_ner_embeddings = hipporag.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)
    phrase_embeddings = hipporag.embed_model.encode_text(oracle_phrases, return_cpu=True, return_numpy=True, norm=True)
    prob_vectors = np.dot(query_ner_embeddings, phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)
    all_phrase_weights, linking_score_map = link_ner_to_node(hipporag, link_top_k, oracle_phrases, prob_vectors, query_ner_list)
    return all_phrase_weights, linking_score_map


def graph_search_with_entities(hipporag: HippoRAG, query_ner_list: list, all_phrase_weights, linking_score_map, query_doc_scores=None):
    """a

    @param hipporag:
    @param query_ner_list:
    @param all_phrase_weights:
    @param linking_score_map:
    @param query_doc_scores: optional, for doc ensemble
    @return:
    """
    # Run Personalized PageRank (PPR) or other Graph Algorithm Doc Scores
    if not hipporag.dpr_only:
        if len(query_ner_list) > 0:
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

            fact_prob = hipporag.facts_to_phrases_mat.dot(ppr_phrase_probs)
            ppr_doc_prob = hipporag.docs_to_facts_mat.dot(fact_prob)
            ppr_doc_prob = min_max_normalize(ppr_doc_prob)
        else:  # dpr_only or no entities found
            ppr_doc_prob = np.ones(len(hipporag.extracted_triples)) / len(hipporag.extracted_triples)

    # Combine Query-Doc and PPR Scores
    if hipporag.doc_ensemble or hipporag.dpr_only:
        # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
        if len(query_ner_list) == 0:
            doc_prob = query_doc_scores
            hipporag.statistics['doc'] = hipporag.statistics.get('doc', 0) + 1
        elif np.min(list(linking_score_map.values())) > hipporag.recognition_threshold:  # high confidence in named entities
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

    if not (hipporag.dpr_only) and len(query_ner_list) > 0:
        # logs
        phrase_one_hop_triples = []
        for phrase_id in np.where(all_phrase_weights > 0)[0]:
            # get all the triples that contain the phrase from hipporag.graph_plus
            for t in list(hipporag.kg_adj_list[phrase_id].items())[:20]:
                phrase_one_hop_triples.append([hipporag.node_phrases[t[0]], t[1]])
            for t in list(hipporag.kg_inverse_adj_list[phrase_id].items())[:20]:
                phrase_one_hop_triples.append([hipporag.node_phrases[t[0]], t[1], 'inv'])

        # get top ranked nodes from doc_prob and hipporag.doc_to_phrases_mat
        nodes_in_retrieved_doc = []
        for doc_id in sorted_doc_ids[:5]:
            node_id_in_doc = list(np.where(hipporag.docs_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
            nodes_in_retrieved_doc.append([hipporag.node_phrases[node_id] for node_id in node_id_in_doc])

        # get top ppr_phrase_probs
        top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]

        # get phrases for top_pagerank_phrase_ids
        top_ranked_nodes = [hipporag.node_phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]
        logs = {'named_entities': query_ner_list, 'linked_node_scores': [list(k) + [float(v)] for k, v in linking_score_map.items()],
                '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                'top_ranked_nodes': top_ranked_nodes, 'nodes_in_retrieved_doc': nodes_in_retrieved_doc}
    else:
        logs = {}

    return logs, sorted_doc_ids, sorted_scores
