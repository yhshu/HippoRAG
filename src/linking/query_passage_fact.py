import difflib
import json

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.hipporag import HippoRAG, get_query_instruction
from src.linking import graph_search_with_entities
from src.linking.dpr_only import dense_passage_retrieval
from src.linking.query_to_fact import link_query_to_fact_core

system_prompt = """Given the following query and potentially relevant facts, your task is: 
- Determine whether these facts are relevant to the query and sufficient to support or answer the query.
- Because some queries require multi-hop reasoning, you may need to integrate information from multiple facts, up to 4 facts.
- Multi-hop reasoning is potentially needed, which requires you to integrate information from multiple facts, up to 4 facts.
- Please select one of the following options as an assessment of relevance: "fully", "partially", or "none".
- If these facts are sufficient to support or answer the query, print all relevant facts.
- Respond in JSON format, starting with rationale, e.g., 

```json
{"thought": "These facts support the query by providing all the necessary information", "relevance": "fully", "facts": [["subject", "predicate", "object"], ["subject", "predicate", "object"]]}
```

or in case of no relevant facts:

```json
{"thought": "None of these facts are relevant to the query.", "relevance": "none", "facts": []}
```"""


def generate_plan(hipporag: HippoRAG, query: str, fact_by_docs: list):
    user_prompt = f"Question: {query}"
    for i, facts in enumerate(fact_by_docs):
        user_prompt += f"\n\nSubgraph {i + 1}: {facts}"
    messages = [SystemMessage(system_prompt),
                HumanMessage(user_prompt)]
    messages = ChatPromptTemplate.from_messages(messages).format_prompt()

    completion = hipporag.client.invoke(messages.to_messages(), temperature=0, response_format={"type": "json_object"})
    contents = completion.content
    try:
        response = json.loads(contents)
        print('relevance', response['relevance'])
        return response
    except Exception as e:
        hipporag.logger.exception(f"Error parsing the response: {e}")
        return {"thought": "", "relevance": "", "facts": []}


def link_by_passage_fact(hipporag: HippoRAG, query: str, link_top_k: int, router=False):
    query_embedding = hipporag.embed_model.encode_text(query, instruction=get_query_instruction(hipporag.embed_model, 'query_to_passage', hipporag.corpus_name),
                                                       return_cpu=True, return_numpy=True, norm=True)
    query_doc_scores = np.dot(hipporag.doc_embedding_mat, query_embedding.T)  # (num_docs, dim) x (1, dim).T = (num_docs, 1)
    query_doc_scores = np.squeeze(query_doc_scores)

    top_doc_idx = np.argsort(query_doc_scores)[-10:][::-1].tolist()

    docs = []
    doc_scores = []
    phrases_by_doc = []
    fact_by_doc = []
    all_candidate_phrases = set()
    for doc_idx in top_doc_idx:
        doc = hipporag.get_passage_by_idx(doc_idx)
        phrases_in_doc = hipporag.get_phrase_in_doc_by_idx(doc_idx)
        docs.append(doc)
        phrases_by_doc.append(phrases_in_doc)
        doc_scores.append(query_doc_scores[doc_idx])
        facts = hipporag.get_triples_and_triple_ids_by_corpus_idx(doc_idx)
        fact_by_doc.append(facts[0])
        for f in facts[0]:
            all_candidate_phrases.update([p.lower() for p in f])

    assert len(docs) == len(fact_by_doc) == len(phrases_by_doc)

    if router:
        plan = generate_plan(hipporag, query, fact_by_doc)
        if plan['relevance'] == 'none':
            sorted_scores = np.sort(query_doc_scores)[::-1][:10]
            logs = {'llm_selected_nodes': []}
            return np.array(top_doc_idx), sorted_scores, logs

        # link the generation to the facts and start graph search
        predicted_nodes = set()
        for fact in plan['facts']:
            if len(fact) > 0:
                predicted_nodes.add(fact[0])
            if len(fact) == 3:
                predicted_nodes.add(fact[2])

        predicted_nodes = list(predicted_nodes)

        all_phrase_weights = np.zeros(len(hipporag.node_phrases))
        linking_score_map = {}
        for phrase in predicted_nodes:
            # choose the most similar phrase from phrases_by_doc
            closest_match = difflib.get_close_matches(phrase, all_candidate_phrases, n=1, cutoff=0)[0]

            phrase_id = hipporag.kb_node_phrase_to_id.get(closest_match, None)
            if phrase_id is None:
                hipporag.logger.error(f'Phrase {phrase} not found in the KG')
                continue
            if hipporag.node_specificity:
                if hipporag.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / hipporag.phrase_to_num_doc[phrase_id]
                all_phrase_weights[phrase_id] = weight
            else:
                all_phrase_weights[phrase_id] = 1.0

            for p in phrases_by_doc:
                if phrase in p:
                    linking_score_map[phrase_id] = doc_scores[phrases_by_doc.index(p)]
                    break

        # graph search
        sorted_doc_ids, sorted_scores, logs = graph_search_with_entities(hipporag, all_phrase_weights, linking_score_map, query_doc_scores=query_doc_scores, damping=None)
        return sorted_doc_ids, sorted_scores, logs

    else:
        candidate_facts = [f for facts in fact_by_doc for f in facts]
        if hipporag.reranker is not None:
            input_facts = candidate_facts
            input_indices = [i for i in range(len(input_facts))]
            candidate_indices, candidate_facts = hipporag.reranker.rerank('fact_reranking', query, input_facts, input_indices, top_k=link_top_k)
            if len(candidate_facts) == 0:
                hipporag.load_dpr_doc_embeddings()
                hipporag.logger.info('No facts found after reranking, return DPR results')
                hipporag.statistics['num_dpr'] += 1
                return dense_passage_retrieval(hipporag, query, False, {})
        facts_to_encode = [str(f) for f in candidate_facts]
        fact_embeddings = hipporag.embed_model.encode_text(facts_to_encode, return_cpu=True, return_numpy=True, norm=True)

        assert len(fact_embeddings) == len(candidate_facts)
        # print(len(fact_embeddings))
        return link_query_to_fact_core(hipporag, query, candidate_facts, fact_embeddings, link_top_k)

