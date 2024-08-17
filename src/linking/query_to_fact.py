import json

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.hipporag import HippoRAG, get_query_instruction
from src.linking import graph_search_with_entities
from src.linking.dpr_only import dense_passage_retrieval, rerank_system_prompt

verify_system_prompt = """Given a query and the top-k retrieved subgraphs from an open Knowledge Graph (KG), evaluate the relevance of these subgraphs to the query. Follow the guidelines below:

- Determine relevance based on whether the subgraphs support or contradict the query. Prioritize supporting facts, especially for questions requiring factual answers.
- If the query involves multi-hop reasoning, consider combining triples from multiple subgraphs to form a coherent rationale.
- After evaluating the subgraphs, generate a rationale for your decision. Then, classify the relevance into one of three categories: "fully", "partially", or "none". Only say relevant when you are confident enough.xx 
- Return in JSON format, e.g., {"thought": "(s1, p1, o1) and (s2, p2, o2) fully support and answer the query", "relevance": "fully"}."""


def link_query_to_fact_core(hipporag: HippoRAG, query, candidate_triples: list, fact_embeddings, link_top_k, graph_search=True,
                            fact_rerank_model_name=None, ppr_doc_verify=False, ppr_phrase_verify=False, merge_dpr=False):
    query_doc_scores = np.zeros(hipporag.docs_to_phrases_mat.shape[0])
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_fact', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    # rank and get link_top_k oracle facts given the query
    query_fact_scores = np.dot(fact_embeddings, query_embedding.T)  # (num_facts, dim) x (1, dim).T = (num_facts, 1)
    query_fact_scores = np.squeeze(query_fact_scores)
    if fact_rerank_model_name is not None:
        # from src.rerank import LLMLogitsReranker
        # reranker = LLMLogitsReranker(fact_rerank_model_name)
        # from src.rerank import RankGPT
        # reranker = RankGPT(rerank_model_name)
        from src.rerank import LLMGenerativeReranker
        reranker = LLMGenerativeReranker(fact_rerank_model_name)
        candidate_fact_indices = np.argsort(query_fact_scores)[-30:][::-1].tolist()
        candidate_facts = [candidate_triples[i] for i in candidate_fact_indices]
        top_k_fact_indicies, top_k_facts = reranker.rerank('fact_reranking', query, candidate_facts, candidate_fact_indices, link_top_k)
        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
        if len(top_k_facts) == 0:
            # return DPR results
            hipporag.load_dpr_doc_embeddings()
            hipporag.logger.info('No facts found after reranking, return DPR results')
            return dense_passage_retrieval(hipporag, query, False)
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
        sorted_doc_ids, sorted_scores, logs, ppr_phrase_probs, ppr_doc_prob = graph_search_with_fact_entities(hipporag, link_top_k, query_doc_scores, query_fact_scores,
                                                                                                              top_k_facts, top_k_fact_indicies, return_ppr=True)

    if fact_rerank_model_name is not None:
        logs['rerank'] = rerank_log

    if ppr_phrase_verify:
        ppr_phrase_probs_top_indices = np.argsort(ppr_phrase_probs)[::-1][:5]
        ppr_top_phrases = [hipporag.node_phrases[i] for i in ppr_phrase_probs_top_indices]
        subgraphs = []
        seen_triples = set()
        for phrase in ppr_top_phrases:
            subgraph = []
            for t in hipporag.triples:
                if (t[0] == phrase or t[2] == phrase) and t not in seen_triples:
                    subgraph.append(t)
                    seen_triples.add(t)
                if len(subgraph) >= 10:
                    break
            if len(subgraph) > 0:
                subgraphs.append(subgraph)

        user_prompt = f"Query: {query}"
        for idx, subgraph in enumerate(subgraphs):
            user_prompt += f"\n\nSubgraph {idx + 1}:\n{subgraph}"
        messages = [SystemMessage(verify_system_prompt), HumanMessage(user_prompt)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        return llm_verify(hipporag, logs, messages, query, sorted_doc_ids, sorted_scores)

    if ppr_doc_verify:
        # verify the subgraphs from the retrieved docs
        fact_by_docs = []
        for doc_idx in sorted_doc_ids[:10]:
            facts, fact_ids = hipporag.get_triples_and_triple_ids_by_corpus_idx(doc_idx)
            fact_by_docs.append(facts)

        user_prompt = f"Query: {query}"
        for idx, facts in enumerate(fact_by_docs):
            user_prompt += f"\n\nSubgraph {idx + 1}:\n{facts}"
        messages = [SystemMessage(verify_system_prompt), HumanMessage(user_prompt)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        return llm_verify(hipporag, logs, messages, query, sorted_doc_ids, sorted_scores)

    if merge_dpr:
        hipporag.load_dpr_doc_embeddings()
        dpr_sorted_doc_ids, dpr_sorted_scores, dpr_logs = dense_passage_retrieval(hipporag, query, False)

        # get dpr docs and HippoRAG docs and let LLM to rerank them
        dpr_docs = []
        merged_sorted_doc_ids = []
        merged_sorted_scores = []
        for doc_id in dpr_sorted_doc_ids[:10]:
            p = hipporag.dataset_df.iloc[doc_id]['paragraph']
            dpr_docs.append(p)
            merged_sorted_doc_ids.append(doc_id)
            merged_sorted_scores.append(dpr_sorted_scores[doc_id])
        hipporag_docs = []
        for doc_id in sorted_doc_ids[:10]:
            p = hipporag.dataset_df.iloc[doc_id]['paragraph']
            if p not in dpr_docs:
                hipporag_docs.append(p)
                merged_sorted_doc_ids.append(doc_id)
                merged_sorted_scores.append(sorted_scores[doc_id])

        candidate_docs = dpr_docs + hipporag_docs
        merged_sorted_doc_ids = np.array(merged_sorted_doc_ids)
        merged_sorted_scores = np.array(merged_sorted_scores)

        user_prompt = f"Query: {query}"
        for idx, doc in enumerate(candidate_docs):
            user_prompt += f"\n\nPassage {idx + 1}:\n{doc}"

        messages = [SystemMessage(rerank_system_prompt), HumanMessage(user_prompt)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        completion = hipporag.client.invoke(messages.to_messages(), temperature=0, response_format={"type": "json_object"})
        contents = completion.content

        try:
            response = json.loads(contents)
            sorted_doc_ids = [merged_sorted_doc_ids[i - 1] for i in response['passage_id']]
            sorted_scores = [merged_sorted_scores[i - 1] for i in response['passage_id']]
            sorted_doc_ids = np.array(sorted_doc_ids)
            sorted_scores = np.array(sorted_scores)
            hipporag.logger.info(f'#candidate passage {len(candidate_docs)}')
        except Exception as e:
            hipporag.logger.exception(f"Error parsing the response: {e}")
            return sorted_doc_ids, sorted_scores, logs

    return sorted_doc_ids, sorted_scores, logs


def llm_verify(hipporag, logs, messages, query, sorted_doc_ids, sorted_scores):
    completion = hipporag.client.invoke(messages.to_messages(), temperature=0, response_format={"type": "json_object"})
    contents = completion.content
    success = False
    try:
        response = json.loads(contents)
        if response['relevance'] == 'fully':
            success = True
    except Exception as e:
        response['relevance'] = 'error'
        hipporag.logger.exception(f"Error parsing the response: {e}")
    logs['verify'] = response
    hipporag.logger.info(response['relevance'])
    if success:
        return sorted_doc_ids, sorted_scores, logs
    else:
        # go back to DPR results
        hipporag.load_dpr_doc_embeddings()
        dpr_sorted_doc_ids, dpr_sorted_scores, dpr_logs = dense_passage_retrieval(hipporag, query, False)
        return dpr_sorted_doc_ids, dpr_sorted_scores, dpr_logs


def graph_search_with_fact_entities(hipporag: HippoRAG, link_top_k: int, query_doc_scores, query_fact_scores, top_k_facts, top_k_fact_indices, return_ppr=False):
    """

    @param hipporag:
    @param link_top_k:
    @param query_doc_scores: Used for DPR or doc ensemble
    @param query_fact_scores: query-fact scores
    @param top_k_facts:
    @param top_k_fact_indices:
    @return:
    """
    all_phrase_weights = np.zeros(len(hipporag.node_phrases))
    linking_score_map = {}
    phrase_scores = {}  # store all fact scores for each phrase

    for rank, f in enumerate(top_k_facts):
        subject_phrase = f[0].lower()
        predicate_phrase = f[1].lower()
        object_phrase = f[2].lower()
        fact_score = query_fact_scores[top_k_fact_indices[rank]]
        for phrase in [subject_phrase, object_phrase]:
            phrase_id = hipporag.kb_node_phrase_to_id.get(phrase, None)
            if phrase_id is not None:
                all_phrase_weights[phrase_id] = 1.0
                if hipporag.node_specificity and hipporag.phrase_to_num_doc[phrase_id] != 0:
                    all_phrase_weights[phrase_id] = 1 / hipporag.phrase_to_num_doc[phrase_id]
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []
            phrase_scores[phrase].append(fact_score)

    # calculate average fact score for each phrase
    for phrase, scores in phrase_scores.items():
        linking_score_map[phrase] = float(np.mean(scores))

    if link_top_k:
        all_phrase_weights, linking_score_map = get_top_k_weights(hipporag, link_top_k, all_phrase_weights, linking_score_map)

    assert sum(all_phrase_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'
    sorted_doc_ids, sorted_scores, logs, ppr_phrase_probs, ppr_doc_prob = graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map,
                                                                                                     query_doc_scores=query_doc_scores, return_ppr=return_ppr)
    if return_ppr is False:
        return sorted_doc_ids, sorted_scores, logs
    return sorted_doc_ids, sorted_scores, logs, ppr_phrase_probs, ppr_doc_prob


def get_top_k_weights(hipporag, link_top_k, all_phrase_weights, linking_score_map):
    # choose top ranked nodes in linking_score_map
    linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
    # only keep the top_k phrases in all_phrase_weights
    top_k_phrases = set(linking_score_map.keys())
    for phrase in hipporag.kb_node_phrase_to_id.keys():
        if phrase not in top_k_phrases:
            phrase_id = hipporag.kb_node_phrase_to_id.get(phrase, None)
            if phrase_id is not None:
                all_phrase_weights[phrase_id] = 0.0
    assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
    return all_phrase_weights, linking_score_map


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
