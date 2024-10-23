import copy

import pandas as pd
from scipy.sparse import csr_array

from processing import *
from glob import glob

import os
import json
from tqdm import tqdm
import pickle
import argparse

from src.data_process.util import check_continuity

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'


def create_graph(dataset: str, extraction_type: str, extraction_model: str, retriever_name: str, threshold: float = 0.9,
                 create_graph_flag: bool = False, cosine_sim_edges: bool = False, passage_node=None):
    corpus = json.load(open(f'data/{dataset}_corpus.json', 'r'))
    processed_retriever_name = retriever_name.replace('/', '_').replace('.', '')
    version = 'v3'
    inter_triple_weight = 1.0
    similarity_max = 1.0
    possible_file_path = f'output/openie_{dataset}_results_{extraction_type}_{extraction_model}_*.json'
    possible_files = glob(possible_file_path)
    assert len(possible_files) > 0, f'No files found for {possible_file_path}'
    if len(possible_files) > 1:
        print(f'[WARN] Note that multiple files found for {possible_file_path}')
    max_samples = np.max([int(file.split('{}_'.format(extraction_model))[1].split('.json')[0]) for file in possible_files])

    extracted_file_path = f'output/openie_{dataset}_results_{extraction_type}_{extraction_model}_{max_samples}.json'
    extracted_file = json.load(open(extracted_file_path, 'r'))

    extracted_triples = extracted_file['docs']
    assert len(extracted_triples) == len(corpus), f'Length of extracted triples {len(extracted_triples)} != length of corpus {len(corpus)}'
    if not extraction_model.startswith('gpt-3.5-turbo'):
        extraction_type = extraction_type + '_' + extraction_model
    phrase_type = 'ents_only_lower_preprocess'  # entities only, lower case, preprocessed
    if cosine_sim_edges:
        graph_type = 'facts_and_sim'  # extracted facts and similar phrases
    else:
        graph_type = 'facts'
    if passage_node is not None:
        graph_type += f'_passage_node_{passage_node}'

    passage_json = []
    phrases = []
    entities = []
    relations = {}  # {(phrase1, phrase2): relation}
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    triples_by_passage = []
    full_neighborhoods = {}
    correct_wiki_format = 0

    for i, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
        document = row['passage']
        raw_ner_entities = row['extracted_entities']
        ner_entities = [processing_phrases(p) for p in row['extracted_entities']]

        triples = row['extracted_triples']

        doc_json = row

        clean_triples = []
        unclean_triples = []
        doc_entities = set()

        # Populate Triples from OpenIE
        for triple in triples:

            triple = [str(s) for s in triple]

            if len(triple) > 1:
                if len(triple) != 3:
                    clean_triple = [processing_phrases(p) for p in triple]

                    incorrectly_formatted_triples.append(triple)
                    unclean_triples.append(triple)
                else:
                    clean_triple = [processing_phrases(p) for p in triple]

                    clean_triples.append(clean_triple)
                    phrases.extend(clean_triple)

                    head_ent = clean_triple[0]
                    tail_ent = clean_triple[2]

                    if head_ent not in ner_entities and tail_ent not in ner_entities:
                        triples_wo_ner_entity.append(triple)

                    relations[(head_ent, tail_ent)] = clean_triple[1]

                    raw_head_ent = triple[0]
                    raw_tail_ent = triple[2]

                    entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_head_ent] = entity_neighborhood

                    entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_tail_ent] = entity_neighborhood

                    for triple_entity in [clean_triple[0], clean_triple[2]]:
                        entities.append(triple_entity)
                        doc_entities.add(triple_entity)

        doc_json['entities'] = list(set(doc_entities))
        doc_json['clean_triples'] = clean_triples
        doc_json['noisy_triples'] = unclean_triples
        triples_by_passage.append(clean_triples)

        passage_json.append(doc_json)

    print('Correct Wiki Format: {} out of {}'.format(correct_wiki_format, len(extracted_triples)))

    try:
        queries_full_df = pd.read_csv(f'output/{dataset}_{extraction_model}_queries.named_entity_output.tsv', sep='\t')

        if 'hotpotqa' in dataset:
            queries = json.load(open(f'data/{dataset}.json', 'r'))
            questions = [q['question'] for q in queries]
            queries_full_df = queries_full_df.set_index('0', drop=False)
        else:
            queries_df = pd.read_json(f'data/{dataset}.json')
            questions = queries_df['question'].values
            queries_full_df = queries_full_df.set_index('question', drop=False)
            queries_full_df = queries_full_df.loc[questions]

        queries_full_df = queries_full_df.loc[questions]
    except Exception as e:
        print('Loading query NER exception: {}'.format(e))
        queries_full_df = pd.DataFrame([], columns=['question', 'triples'])

    q_entities = []
    q_entities_by_doc = []
    for doc_ents in tqdm(queries_full_df.triples):
        try:
            doc_ents = eval_json_str(doc_ents).get('named_entities', [])
            clean_doc_ents = [processing_phrases(p) for p in doc_ents]
        except:
            print("No named entities found for one query")
            clean_doc_ents = []
        q_entities.extend(clean_doc_ents)
        q_entities_by_doc.append(clean_doc_ents)
    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ['equivalent'])
    q_phrases = list(np.unique(q_entities))
    all_phrases = copy.deepcopy(unique_phrases)
    all_phrases.extend(q_phrases)

    kb_df = pd.DataFrame(unique_phrases, columns=['strings'])
    kb_df2 = copy.deepcopy(kb_df)
    kb_df['type'] = 'query'
    kb_df2['type'] = 'kb'
    kb_full_df = pd.concat([kb_df, kb_df2])
    kb_full_df.to_csv('output/kb_to_kb.tsv', sep='\t')

    rel_kb_df = pd.DataFrame(unique_relations, columns=['strings'])
    rel_kb_df2 = copy.deepcopy(rel_kb_df)
    rel_kb_df['type'] = 'query'
    rel_kb_df2['type'] = 'kb'
    rel_kb_full_df = pd.concat([rel_kb_df, rel_kb_df2])
    rel_kb_full_df.to_csv('output/rel_kb_to_kb.tsv', sep='\t')

    query_df = pd.DataFrame(q_phrases, columns=['strings'])
    query_df['type'] = 'query'
    kb_df['type'] = 'kb'
    kb_query_df = pd.concat([kb_df, query_df])
    kb_query_df.to_csv('output/query_to_kb.tsv', sep='\t')

    if create_graph_flag:
        print('Creating Graph')

        node_json = [{'idx': i, 'name': p} for i, p in enumerate(unique_phrases)]
        kb_phrase_to_id_dict = {p: i for i, p in enumerate(unique_phrases)}
        assert len(node_json) == len(kb_phrase_to_id_dict)
        print('Number of phrase nodes: {}'.format(len(node_json)))

        if passage_node is not None:
            len_node = len(node_json)
            # add all passages to node_json and kb_phrase_to_id_dict
            passage_node_idx = 0
            for i, doc in enumerate(passage_json):
                p = doc['passage']
                if p not in kb_phrase_to_id_dict:
                    node_idx = len_node + passage_node_idx
                    passage_node_idx += 1
                    node_json.append({'idx': node_idx, 'name': p})
                    kb_phrase_to_id_dict[p] = node_idx
                    assert node_idx < len(node_json), f'Node idx {node_idx} not smaller than length of nodes {len(node_json)}'

        print('Number of passage nodes: {}'.format(len(node_json) - len(unique_phrases)))
        extracted_triples = []

        for triples in triples_by_passage:
            extracted_triples.extend([tuple(t) for t in triples])

        triplet_fact_to_id_dict = {f: i for i, f in enumerate(extracted_triples)}
        fact_json_list = [{'idx': i, 'head': t[0], 'relation': t[1], 'tail': t[2]} for i, t in enumerate(extracted_triples)]

        json.dump(passage_json, open('output/{}_{}_graph_passage_chatgpt_openIE.{}_{}.{}.subset.json'.format(dataset, graph_type, phrase_type, extraction_type, version), 'w'))
        # node_json: [{'idx': 0, 'name': 'phrase1'}, ...]
        json.dump(node_json, open('output/{}_{}_graph_nodes_chatgpt_openIE.{}_{}.{}.subset.json'.format(dataset, graph_type, phrase_type, extraction_type, version), 'w'))
        # fact_json_list: [{'idx': 0, 'head': 'phrase1', 'relation': 'relation1', 'tail': 'phrase2'}, ...]
        json.dump(fact_json_list, open('output/{}_{}_graph_clean_facts_chatgpt_openIE.{}_{}.{}.subset.json'.format(
            dataset, graph_type, phrase_type, extraction_type, version), 'w'))

        # kb_phrase_to_id_dict: {'phrase1': 0, 'phrase2': 1, ...}
        pickle.dump(kb_phrase_to_id_dict, open('output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        # triplet_fact_to_id_dict: {('phrase1', 'relation1', 'phrase2'): 0, ...}
        pickle.dump(triplet_fact_to_id_dict, open('output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        graph_json = {}  # {phrase: {phrase2: ('triple'/'similarity', frequency/score)}}

        docs_to_facts = {}  # {(doc id, fact id) -> frequency}
        facts_to_phrases = {}  # {(fact id, phrase id) -> frequency}
        graph = {}  # {(phrase id, phrase id) -> frequency}

        num_triple_edges = 0

        # Creating Adjacency and Document to Phrase Matrices
        for doc_id, triples in tqdm(enumerate(triples_by_passage), total=len(triples_by_passage)):

            doc_phrases = []
            fact_edges = []

            # Iterate over triples
            for triple in triples:
                triple = tuple(triple)

                fact_id = triplet_fact_to_id_dict[triple]

                if len(triple) == 3:
                    relation = triple[1]
                    triple = np.array(triple)[[0, 2]]

                    docs_to_facts[(doc_id, fact_id)] = 1

                    for i, phrase in enumerate(triple):
                        phrase_id = kb_phrase_to_id_dict[phrase]
                        doc_phrases.append(phrase_id)

                        facts_to_phrases[(fact_id, phrase_id)] = 1

                        for phrase2 in triple[i + 1:]:
                            phrase2_id = kb_phrase_to_id_dict[phrase2]

                            fact_edge_r = (phrase_id, phrase2_id)
                            fact_edge_l = (phrase2_id, phrase_id)

                            fact_edges.append(fact_edge_r)
                            fact_edges.append(fact_edge_l)

                            graph[fact_edge_r] = graph.get(fact_edge_r, 0.0) + inter_triple_weight
                            graph[fact_edge_l] = graph.get(fact_edge_l, 0.0) + inter_triple_weight

                            phrase_edges = graph_json.get(phrase, {})
                            edge = phrase_edges.get(phrase2, ('triple', 0))
                            phrase_edges[phrase2] = ('triple', edge[1] + 1)
                            graph_json[phrase] = phrase_edges

                            phrase_edges = graph_json.get(phrase2, {})
                            edge = phrase_edges.get(phrase, ('triple', 0))
                            phrase_edges[phrase] = ('triple', edge[1] + 1)
                            graph_json[phrase2] = phrase_edges

                            num_triple_edges += 1

        pickle.dump(docs_to_facts, open('output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        pickle.dump(facts_to_phrases, open('output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        docs_to_facts_mat = csr_array(([int(v) for v in docs_to_facts.values()], ([int(e[0]) for e in docs_to_facts.keys()], [int(e[1]) for e in docs_to_facts.keys()])),
                                      shape=(len(triples_by_passage), len(extracted_triples)))
        facts_to_phrases_mat = csr_array(([int(v) for v in facts_to_phrases.values()], ([e[0] for e in facts_to_phrases.keys()], [e[1] for e in facts_to_phrases.keys()])),
                                         shape=(len(extracted_triples), len(unique_phrases)))

        assert docs_to_facts_mat.shape[0] == len(corpus), f"docs_to_facts_mat.shape[0] {docs_to_facts_mat.shape[0]} != len(corpus) {len(corpus)}"

        pickle.dump(docs_to_facts_mat, open('output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        pickle.dump(facts_to_phrases_mat,
                    open('output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        pickle.dump(graph, open('output/{}_{}_graph_fact_doc_edges_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        print('Loading Vectors')

        # Expanding OpenIE triples with cosine similarity-based synonymy edges
        if cosine_sim_edges:
            if 'colbert' in retriever_name:
                kb_similarity = pickle.load(open('data/lm_vectors/colbert/nearest_neighbor_kb_to_kb.p'.format(processed_retriever_name), 'rb'))
            else:
                kb_similarity = pickle.load(open('data/lm_vectors/{}_mean/nearest_neighbor_kb_to_kb.p'.format(processed_retriever_name), 'rb'))

            print('Augmenting Graph from Similarity')

            graph_with_synonym = copy.deepcopy(graph)  # {(phrase id, phrase id): frequency/score}

            kb_similarity = {processing_phrases(k): v for k, v in kb_similarity.items()}

            synonym_candidates = []  # [(phrase, [(synonym, score), ...]), ...]

            for phrase in tqdm(kb_similarity.keys(), total=len(kb_similarity)):

                synonyms = []

                if len(re.sub('[^A-Za-z0-9]', '', phrase)) > 2:
                    phrase_id = kb_phrase_to_id_dict.get(phrase, None)

                    if phrase_id is not None:

                        nns = kb_similarity[phrase]

                        num_nns = 0
                        for nn, score in zip(nns[0], nns[1]):
                            nn = processing_phrases(nn)
                            if score < threshold or num_nns > 100:
                                break

                            if nn != phrase:

                                phrase2_id = kb_phrase_to_id_dict.get(nn)

                                if phrase2_id is not None:
                                    phrase2 = nn

                                    sim_edge = (phrase_id, phrase2_id)
                                    synonyms.append((nn, score))

                                    relations[(phrase, phrase2)] = 'equivalent'
                                    graph_with_synonym[sim_edge] = similarity_max * score

                                    num_nns += 1

                                    phrase_edges = graph_json.get(phrase, {})
                                    edge = phrase_edges.get(phrase2, ('similarity', 0))
                                    if edge[0] == 'similarity':
                                        phrase_edges[phrase2] = ('similarity', edge[1] + score)
                                        graph_json[phrase] = phrase_edges

                synonym_candidates.append((phrase, synonyms))

            pickle.dump(synonym_candidates, open(
                'output/{}_similarity_edges_mean_{}_thresh_{}_{}_{}.{}.subset.p'.format(dataset, threshold, phrase_type, extraction_type, processed_retriever_name, version), 'wb'))
        else:
            graph_with_synonym = graph

        if passage_node is not None:
            # add edges between phrases and passages
            for i, doc in enumerate(passage_json):
                p = doc['passage']
                p_id = kb_phrase_to_id_dict[p]
                for phrase in doc['entities']:
                    phrase_id = kb_phrase_to_id_dict[phrase]
                    graph_with_synonym[(p_id, phrase_id)] = 1.0
                    phrase_edges = graph_json.get(p, {})
                    edge = phrase_edges.get(phrase, ('passage_has', 0))
                    phrase_edges[phrase] = ('passage_has', edge[1] + 1)
                    graph_json[p] = phrase_edges
                    relations[(p, phrase)] = 'passage_has'

                    if passage_node == 'bidirectional':
                        graph_with_synonym[(phrase_id, p_id)] = 1.0
                        phrase_edges = graph_json.get(phrase, {})
                        edge = phrase_edges.get(p, ('in_passage', 0))
                        phrase_edges[p] = ('in_passage', edge[1] + 1)
                        graph_json[phrase] = phrase_edges
                        relations[(phrase, p)] = 'in_passage'

        pickle.dump(relations,
                    open('output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, processed_retriever_name, version),
                         'wb'))

        print('Saving Graph')

        synonymy_edges = set([edge for edge in relations.keys() if relations[edge] == 'equivalent'])
        passage_edges = set([edge for edge in relations.keys() if relations[edge] in ['passage_has', 'in_passage']])

        statistics_df = [('Total Phrases', len(phrases)),
                         ('Unique Phrases', len(unique_phrases)),
                         ('Number of Passages', len(passage_json)),
                         ('Number of Individual Triples', len(extracted_triples)),
                         ('Number of Incorrectly Formatted Triples (LLM Error)', len(incorrectly_formatted_triples)),
                         ('Number of Triples w/o NER Entities (LLM Error)', len(triples_wo_ner_entity)),
                         ('Number of Unique Individual Triples', len(triplet_fact_to_id_dict)),
                         ('Number of Entities', len(entities)),
                         ('Number of Relations', len(relations)),
                         ('Number of Unique Entities', len(np.unique(entities))),
                         ('Number of Synonymy Edges', len(synonymy_edges)),
                         ('Number of Passage Edges', len(passage_edges)),
                         ('Number of Unique Relations', len(unique_relations))]

        print(pd.DataFrame(statistics_df).set_index(0))

        if similarity_max == 1.0:
            pickle.dump(graph_with_synonym, open(
                'output/{}_{}_graph_mean_{}_thresh_{}_{}_{}.{}.subset.p'.format(dataset, graph_type, threshold, phrase_type,
                                                                                extraction_type, processed_retriever_name, version), 'wb'))
        else:
            pickle.dump(graph_with_synonym, open(
                'output/{}_{}_graph_mean_{}_thresh_{}_{}_sim_max_{}_{}.{}.subset.p'.format(dataset, graph_type, threshold,
                                                                                           phrase_type, extraction_type, similarity_max, processed_retriever_name, version), 'wb'))

        json.dump(graph_json, open('output/{}_{}_graph_chatgpt_openIE.{}_{}.{}.subset.json'.format(dataset, graph_type, phrase_type,
                                                                                                   extraction_type, version), 'w'))


if __name__ == '__main__':
    # Get the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--extraction_model', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--create_graph', action='store_true')
    parser.add_argument('--extraction_type', type=str)
    parser.add_argument('--cosine_sim_edges', action='store_true')

    args = parser.parse_args()
    dataset = args.dataset
    retriever_name = args.model_name
    extraction_model = args.extraction_model.replace('/', '_')
    threshold = args.threshold
    create_graph_flag = args.create_graph
    extraction_type = args.extraction_type
    cosine_sim_edges = args.cosine_sim_edges

    create_graph(dataset, extraction_type, extraction_model, retriever_name, threshold, create_graph_flag, cosine_sim_edges)
