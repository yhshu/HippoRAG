import json
import logging
import os
import _pickle as pickle
from collections import defaultdict
from glob import glob
from typing import List, Dict

import igraph as ig
import numpy as np
import pandas as pd
import torch
from networkx.classes.filters import hide_nodes
from scipy.sparse import csr_matrix

from tqdm import tqdm

from src.langchain_util import init_langchain_model, LangChainModel
from src.lm_wrapper import EmbeddingModelWrapper
from src.lm_wrapper.gritlm import GritLMWrapper
from src.lm_wrapper.sentence_transformers_util import SentenceTransformersWrapper
from src.lm_wrapper.util import init_embedding_model
from src.named_entity_extraction_parallel import named_entity_recognition
from src.processing import processing_phrases, softmax_with_zeros, eval_json_str, min_max_normalize

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

COLBERT_CKPT_DIR = "exp/colbertv2.0"


def get_query_instruction(embedding_model: EmbeddingModelWrapper, task=None, dataset_name=None):
    if isinstance(embedding_model, GritLMWrapper):
        if task == 'ner_to_node':
            return 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.'
        elif task == 'query_to_node':
            if 'scifact' in dataset_name.lower():
                return 'Given a claim, retrieve relevant phrases that are mentioned in this claim.'
            return 'Given a question, retrieve relevant phrases that are mentioned in this question.'
        elif task == 'query_to_fact':
            if 'scifact' in dataset_name.lower():
                return 'Given a claim, retrieve relevant triplet facts that support or refute the claim.'
            return 'Given a question, retrieve relevant triplet facts that matches this question.'
        elif task == 'query_to_sentence':
            if 'scifact' in dataset_name.lower():
                return 'Given a claim, retrieve relevant sentences that support or refute the claim.'
            return 'Given a question, retrieve relevant sentences that best answer the question.'
        elif task is None or task == 'query_to_passage':
            if 'scifact' in dataset_name.lower():
                return 'Given a scientific claim, retrieve documents that support or refute the claim.'
            else:
                return 'Given a question, retrieve relevant documents that best answer the question.'
        else:
            print('Task instruction not found for {}'.format(task))
            return ''
    elif isinstance(embedding_model, SentenceTransformersWrapper):
        if embedding_model.model_name.startswith('Alibaba-NLP/gte-Qwen'):
            return 'query'
    return None


class HippoRAG:

    def __init__(self, corpus_name='hotpotqa', extraction_model='openai', extractor_name='gpt-3.5-turbo-1106',
                 graph_creating_retriever_name='facebook/contriever', extraction_type='ner', graph_type='facts_and_sim',
                 sim_threshold=0.8, node_specificity=True,
                 doc_ensemble=False, colbert_config=None, dpr_only=False,
                 graph_alg='ppr', damping=0.1, recognition_threshold=0.9, corpus_path=None,
                 qa_model: LangChainModel = None, linker_name=None, reranker_name=None, directed_graph=False):
        """
        @param corpus_name: Name of the dataset to use for retrieval
        @param extraction_model: LLM provider for query NER, e.g., 'openai' or 'together'
        @param extractor_name: LLM name used for query NER
        @param graph_creating_retriever_name: Retrieval encoder used to link query named entities with query nodes
        @param extraction_type: Type of NER extraction during indexing
        @param graph_type: Type of graph used by HippoRAG
        @param sim_threshold: Synonymy threshold which was used to create the graph that will be used by HippoRAG
        @param node_specificity: Flag that determines whether node specificity will be used
        @param doc_ensemble: Flag to determine whether to use uncertainty-based ensemble
        @param colbert_config: ColBERTv2 configuration
        @param dpr_only: Flag to determine whether HippoRAG will be used at all
        @param graph_alg: Type of graph algorithm to be used for retrieval, defaults ot PPR
        @param damping: Damping factor for PPR
        @param recognition_threshold: Threshold used for uncertainty-based ensembling.
        @param corpus_path: path to the corpus file (see the format in README.md), not needed for now if extraction files are already present
        @param qa_model: QA model
        @param linker_name: linking model name
        @param reranker_name: reranker model name, for reranking task during linking, choices: openai model names, 'llama_cpp_server', local llama 3 lora path
        """

        self.corpus_name = corpus_name
        self.extraction_model_name = extractor_name
        self.extraction_model_name_processed = extractor_name.replace('/', '_')
        self.client = init_langchain_model(extraction_model, extractor_name)
        assert graph_creating_retriever_name
        if linker_name is None:
            linker_name = graph_creating_retriever_name
        self.graph_creating_retriever_name = graph_creating_retriever_name  # 'colbertv2', 'facebook/contriever', or other HuggingFace models
        self.graph_creating_retriever_name_processed = graph_creating_retriever_name.replace('/', '_').replace('.', '')
        self.linking_retriever_name = linker_name
        self.linking_retriever_name_processed = linker_name.replace('/', '_').replace('.', '')

        self.extraction_type = extraction_type
        self.graph_type = graph_type
        self.phrase_type = 'ents_only_lower_preprocess'
        self.sim_threshold = sim_threshold
        self.node_specificity = node_specificity
        if colbert_config is None:
            self.colbert_config = {'root': f'data/lm_vectors/colbert/{corpus_name}',
                                   'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
        else:
            self.colbert_config = colbert_config  # a dict, 'root', 'doc_index_name', 'phrase_index_name'

        self.graph_alg = graph_alg
        self.damping = damping
        self.recognition_threshold = recognition_threshold

        self.version = 'v3'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('HippoRAG')

        try:
            query_ner_path = f'output/{self.corpus_name}_{self.extraction_model_name_processed}_queries.named_entity_output.tsv'
            self.query_ner_cache = pd.read_csv(query_ner_path, sep='\t')
            self.logger.info('Loaded query NER cache from: ' + query_ner_path)
        except Exception as e:
            self.query_ner_cache = pd.DataFrame([], columns=['query', 'triples'])
            self.logger.error('Query NER cache is not loaded ' + str(e))

        query_key = 'query' if 'query' in self.query_ner_cache else 'question'
        self.query_ner_cache = {row[query_key]: eval_json_str(row['triples']) for i, row in
                                self.query_ner_cache.iterrows()}

        self.embed_model = init_embedding_model(self.linking_retriever_name)
        self.dpr_only = dpr_only
        self.doc_ensemble = doc_ensemble
        self.corpus_path = corpus_path
        self.is_directed_graph = directed_graph
        self.node_to_doc_strategy = 'multiply'  # 'multiply', 'map'

        # Loading Important Corpus Files
        if not self.dpr_only:
            self.load_index_files()

            # Construct Graph
            self.build_graph()

            # Loading Node Embeddings
            self.load_node_vectors()
        self.load_corpus()
        self.triple_embeddings = None
        self.doc_embedding_mat = None

        if (doc_ensemble or dpr_only) and self.linking_retriever_name not in ['colbertv2', 'bm25']:
            # Loading Doc Embeddings
            self.load_dpr_doc_embeddings()

        if self.linking_retriever_name == 'colbertv2':
            from colbert import Searcher
            from colbert.infra import RunConfig, Run, ColBERTConfig
            from src.colbertv2_indexing import colbertv2_index

            if self.dpr_only is False or self.doc_ensemble:
                colbertv2_index(self.node_phrases.tolist(), self.corpus_name, 'phrase',
                                self.colbert_config['phrase_index_name'], overwrite=True)
                with Run().context(RunConfig(nranks=1, experiment="phrase", root=self.colbert_config['root'])):
                    config = ColBERTConfig(root=self.colbert_config['root'], )
                    self.phrase_searcher = Searcher(index=self.colbert_config['phrase_index_name'], config=config, verbose=0)
            if self.doc_ensemble or dpr_only:
                colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus',
                                self.colbert_config['doc_index_name'], overwrite=True)
                with Run().context(RunConfig(nranks=1, experiment="corpus", root=self.colbert_config['root'])):
                    config = ColBERTConfig(root=self.colbert_config['root'], )
                    self.corpus_searcher = Searcher(index=self.colbert_config['doc_index_name'], config=config, verbose=0)

        self.statistics = defaultdict(int)
        self.ensembling_debug = []
        if qa_model is None:
            qa_model = LangChainModel('openai', 'gpt-3.5-turbo')
        self.qa_model = init_langchain_model(qa_model.provider, qa_model.model_name)

        self.reranker = None
        self.reranker_name = None
        if reranker_name is not None:
            if (reranker_name.startswith('gpt') or reranker_name.startswith('ft:gpt')
                    or reranker_name.startswith('o1-') or reranker_name in ['llama_cpp_server']):
                # from src.rerank import LLMLogitsReranker
                # reranker = LLMLogitsReranker(fact_rerank_model_name)
                # from src.rerank import RankGPT
                # reranker = RankGPT(rerank_model_name)
                from src.rerank import LLMGenerativeReranker
                self.reranker = LLMGenerativeReranker(reranker_name)
            elif reranker_name in ['oracle_triple']:
                from src.rerank import OracleTripleFiltering
                self.reranker = OracleTripleFiltering(reranker_name)
            else:  # load Llama 3.1 model with LoRA
                from src.rerank import HFLoRAModelGenerativeReranker
                self.reranker = HFLoRAModelGenerativeReranker(reranker_name)
            self.reranker_name = reranker_name
            self.statistics['num_dpr'] = 0

    def get_passage_by_idx(self, passage_idx):
        """
        Get the passage by its index
        @param passage_idx: the index of the passage.
        @return: the passage.
        """
        return self.dataset_df.iloc[passage_idx]['paragraph']

    def get_raw_extraction_by_passage_idx(self, passage_idx, chunk=False):
        """
        Get the extraction results for a specific passage.
        @param passage_idx: the passage idx, i.e., 'idx' within each passage dict, not the array index for the corpus
        @param chunk: whether the corpus is chunked
        @return: the extraction results for the passage
        """
        # find item with idx == passage_idx in self.extracted_triples
        for item in self.extracted_triples:
            if not chunk and item['idx'] == passage_idx:
                return item
            elif chunk and (item['idx'] == passage_idx or item['idx'].startswith(passage_idx + '_')):
                return item
        return None

    def get_triples_and_triple_ids_by_corpus_idx(self, corpus_idx):
        """
        Get the triples in the knowledge graph for a specific passage.
        @param corpus_idx: the passage idx, i.e., 'idx' within each passage dict, not the array index for the corpus
        @return: the triples in the knowledge graph for the passage and their triple ids
        """
        # Get the start and end indices for the rows corresponding to the corpus_idx
        start_idx = self.docs_to_triples_mat.indptr[corpus_idx]
        end_idx = self.docs_to_triples_mat.indptr[corpus_idx + 1]

        # Extract the triple ids and corresponding triples from the matrix
        triple_ids = self.docs_to_triples_mat.indices[start_idx:end_idx]
        triples = [self.triplet_id_to_triple_dict[triple_id] for triple_id in triple_ids]

        return triples, triple_ids

    def get_triples_and_triple_ids_by_passage_content(self, passage_content):
        """

        Get the triples in the knowledge graph for a specific passage.
        """
        corpus_idx = self.get_corpus_idx_by_passage_content(passage_content)
        assert corpus_idx is not None, 'Passage not found in the corpus'
        return self.get_triples_and_triple_ids_by_corpus_idx(corpus_idx)

    def get_corpus_idx_by_passage_idx(self, passage_idx):
        """
        Get the corpus index by the passage index
        @param passage_idx: the passage index
        @return: the corpus index
        """
        for corpus_idx, item in enumerate(self.corpus):
            if item['idx'] == passage_idx:
                return corpus_idx
        return None

    def get_corpus_idx_by_passage_content(self, passage_content):
        """
        Get the corpus index by the passage content
        @param passage_content: the passage content
        @return: the corpus index
        """
        for corpus_idx, item in enumerate(self.corpus):
            if (item['title'] + '\n' + item['text']) == passage_content:
                return corpus_idx
        return None

    def get_triple(self, triple_idx):
        if triple_idx < len(self.triples):
            return self.triples[triple_idx]
        else:
            self.logger.error(f'Triple ID {triple_idx} not found in the graph')
            return None

    def get_shortest_distance_between_nodes(self, node1: str, node2: str):
        """
        Get the shortest distance between two nodes in the graph
        @param node1: node1 phrase
        @param node2: node2 phrase
        @return: the shortest distance between the two nodes
        """
        try:
            node1_id = np.where(self.node_phrases == node1)[0][0]
            node2_id = np.where(self.node_phrases == node2)[0][0]

            return self.g.shortest_paths(node1_id, node2_id)[0][0]
        except Exception as e:
            return -1

    def convert_ppr_node_to_doc_score(self, ppr_node_probs):
        num_phrases = len(ppr_node_probs)
        if self.node_to_doc_strategy == 'multiply':
            fact_prob = self.triples_to_phrases_mat.dot(ppr_node_probs)
            ppr_doc_prob = self.docs_to_triples_mat.dot(fact_prob)
        elif self.node_to_doc_strategy == 'map':
            from scipy.sparse import csr_matrix
            r_diag_csr = csr_matrix((ppr_node_probs, (np.arange(num_phrases), np.arange(num_phrases))), shape=(num_phrases, num_phrases))
            edge_importance_csr = r_diag_csr.dot(self.normalized_transition_mat)
            symetric_edge_importance_csr = edge_importance_csr + edge_importance_csr.T
            new_fact_importance = symetric_edge_importance_csr[self.edge_source_nodes, self.edge_target_nodes]
            ppr_doc_prob = self.docs_to_triples_mat.dot(new_fact_importance.T)[:, 0]
        else:
            raise ValueError('Invalid edge_to_fact_strategy')
        return min_max_normalize(ppr_doc_prob)

    def rank_docs(self, query: str, doc_top_k=10, link_top_k=3, linking='ner_to_node', oracle_triples=None):
        """
        Rank documents based on the query
        @param query: the input phrase
        @param doc_top_k: the number of documents to return
        @param link_top_k: the number of top-k items to retrieve
        @param linking: the linking method to use: 'ner_to_node', 'query_to_node', 'query_to_fact'
        @param oracle_triples: the oracle extraction results, used for upper bound evaluation
        @return: the ranked document ids and their scores
        """

        assert isinstance(query, str), 'Query must be a string'
        query_doc_scores = None

        if oracle_triples is not None and len(oracle_triples) == 0:
            self.logger.info('No oracle triples found for the query: ' + query)

        if self.dpr_only:
            from src.linking.dpr_only import dense_passage_retrieval
            sorted_doc_ids, sorted_scores, logs = dense_passage_retrieval(self, query)
            return sorted_doc_ids.tolist()[:doc_top_k], sorted_scores.tolist()[:doc_top_k], logs

        elif oracle_triples and linking == 'ner_to_node':
            from src.linking.ner_to_node import oracle_ner_to_node, graph_search_with_entities
            query_ner_list = self.query_ner(query)
            oracle_phrases = set()
            for t in oracle_triples:
                if len(t):
                    oracle_phrases.add(t[0])
                if len(t) >= 3:
                    oracle_phrases.add(t[2])
            oracle_phrases = list(oracle_phrases)
            all_phrase_weights, linking_score_map = oracle_ner_to_node(self, query_ner_list, oracle_phrases, link_top_k)
            sorted_doc_ids, sorted_scores, doc_rank_logs = graph_search_with_entities(self, query_ner_list, all_phrase_weights, linking_score_map, query_doc_scores)

        elif oracle_triples and linking == 'query_to_node':
            from src.linking import graph_search_with_entities

            oracle_node_phrases = set()
            for t in oracle_triples:
                if len(t):
                    oracle_node_phrases.add(t[0])
                if len(t) >= 3:
                    oracle_node_phrases.add(t[2])
            oracle_node_phrases = list(oracle_node_phrases)
            node_embeddings = self.embed_model.encode_text(oracle_node_phrases, return_cpu=True, return_numpy=True, norm=True)
            query_embedding = self.embed_model.encode_text(query, instruction=get_query_instruction(self.embed_model, 'query_to_node', self.corpus_name),
                                                           return_cpu=True, return_numpy=True, norm=True)
            # rank and get link_top_k oracle nodes given the query
            query_node_scores = np.dot(node_embeddings,
                                       query_embedding.T)  # (num_nodes, dim) x (1, dim).T = (num_nodes, 1)
            query_node_scores = np.squeeze(query_node_scores)
            if link_top_k:
                top_k_indices = np.argsort(query_node_scores)[-link_top_k:][::-1].tolist()
            else:
                top_k_indices = np.argsort(query_node_scores)[::-1].tolist()

            top_k_phrases = [oracle_node_phrases[i] for i in top_k_indices]

            all_phrase_weights = np.zeros(len(self.node_phrases))
            for i, phrase in enumerate(self.node_phrases):
                matching_index = next(
                    (index for index, top_phrase in enumerate(top_k_phrases) if phrase.lower() == top_phrase.lower()),
                    None)
                if matching_index is not None:
                    all_phrase_weights[i] = query_node_scores[matching_index]

            if sum(all_phrase_weights) == 0:
                sorted_doc_ids, sorted_scores, doc_rank_logs = self.query_to_node_linking(link_top_k, query, query_doc_scores)
            else:
                all_phrase_weights = softmax_with_zeros(all_phrase_weights)
                linking_score_map = {oracle_node_phrases[i]: query_node_scores[i] for i in top_k_indices}
                sorted_doc_ids, sorted_scores, doc_rank_logs = graph_search_with_entities(self, all_phrase_weights, linking_score_map, None)

        elif oracle_triples and linking == 'query_to_fact' and self.reranker_name != 'oracle_triple':
            from src.linking.query_to_fact import oracle_query_to_fact
            sorted_doc_ids, sorted_scores, logs = oracle_query_to_fact(self, query, oracle_triples, link_top_k)
            return sorted_doc_ids.tolist()[:doc_top_k], sorted_scores.tolist()[:doc_top_k], logs

        elif linking == 'ner_to_node':
            from src.linking.ner_to_node import link_node_by_dpr, link_node_by_colbertv2, graph_search_with_entities
            all_phrase_weights = np.zeros(len(self.node_phrases))
            linking_score_map = {}
            query_ner_list = self.query_ner(query)
            if 'colbertv2' in self.linking_retriever_name:
                from colbert.data import Queries
                queries = Queries(path=None, data={0: query})
                if self.doc_ensemble:
                    query_doc_scores = np.zeros(self.docs_to_phrases_mat.shape[0])
                    ranking = self.corpus_searcher.search_all(queries, k=self.docs_to_phrases_mat.shape[0])
                    # max_query_score = self.get_colbert_max_score(query)
                    for corpus_id, rank, score in ranking.data[0]:
                        query_doc_scores[corpus_id] = score
                if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
                    all_phrase_weights, linking_score_map = link_node_by_colbertv2(self, query_ner_list, link_top_k)
            else:  # huggingface dense retrieval
                if self.doc_ensemble:
                    query_embedding = self.embed_model.encode_text(query, instruction=get_query_instruction(self.embed_model, 'ner_to_node', self.corpus_name),
                                                                   return_cpu=True, return_numpy=True, norm=True)
                    query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                    query_doc_scores = query_doc_scores.T[0]

                if len(query_ner_list) > 0:  # if no entities are found, assign uniform probability to documents
                    all_phrase_weights, linking_score_map = link_node_by_dpr(self, query_ner_list, link_top_k)

            if sum(all_phrase_weights) == 0:
                all_phrase_weights = np.ones(len(self.node_phrases)) / len(self.node_phrases)
            sorted_doc_ids, sorted_scores, doc_rank_logs = graph_search_with_entities(self, query_ner_list, all_phrase_weights, linking_score_map, query_doc_scores)

        elif linking == 'query_to_node' or (oracle_triples is not None and len(oracle_triples) == 0):
            from src.linking.query_to_node import link_node_by_dpr
            from src.linking import graph_search_with_entities
            if 'colbertv2' in self.linking_retriever_name:
                pass  # todo
            else:
                all_phrase_weights, linking_score_map = link_node_by_dpr(self, query, top_k=link_top_k)
            sorted_doc_ids, sorted_scores, doc_rank_logs = graph_search_with_entities(self, all_phrase_weights, linking_score_map, query_doc_scores)
            return sorted_doc_ids.tolist()[:doc_top_k], sorted_scores.tolist()[:doc_top_k], doc_rank_logs

        elif linking == 'query_to_fact':
            from src.linking.query_to_fact import link_fact_by_dpr
            if 'colbertv2' in self.linking_retriever_name:
                pass
            else:  # huggingface dense retrieval
                self.load_triple_vectors()
                sorted_doc_ids, sorted_scores, log = link_fact_by_dpr(self, query, link_top_k=link_top_k, oracle_triples=oracle_triples)
                return sorted_doc_ids.tolist()[:doc_top_k], sorted_scores.tolist()[:doc_top_k], log

        elif linking == 'query_to_passage':
            from src.linking.query_to_passage import link_by_passage_fact
            if 'colbertv2' in self.linking_retriever_name:
                pass
            else:
                self.load_dpr_doc_embeddings()
                sorted_doc_ids, sorted_scores, doc_rank_logs = link_by_passage_fact(self, query, link_top_k=link_top_k)

        elif linking == 'query_passage_node':
            self.load_dpr_doc_embeddings()
            from src.linking.query_passage_node import link_by_passage_node
            sorted_doc_ids, sorted_scores, doc_rank_logs = link_by_passage_node(self, query, link_top_k=link_top_k)

        elif linking == 'query_passage_fact':
            self.load_dpr_doc_embeddings()
            from src.linking.query_passage_fact import link_by_passage_fact
            sorted_doc_ids, sorted_scores, doc_rank_logs = link_by_passage_fact(self, query, link_top_k=link_top_k)

        else:
            raise NotImplementedError('Linking method not implemented')

        return sorted_doc_ids.tolist()[:doc_top_k], sorted_scores.tolist()[:doc_top_k], doc_rank_logs

    def query_ner(self, query):
        if self.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities
            try:
                if query in self.query_ner_cache:
                    query_ner_list = self.query_ner_cache[query]['named_entities']
                else:
                    query_ner_json, total_tokens = named_entity_recognition(self.client, query)
                    query_ner_list = eval_json_str(query_ner_json).get('named_entities', [])

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except:
                self.logger.error('Error in Query NER')
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector, max_depth=1):

        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def load_corpus(self):
        if self.corpus_path is None:
            self.corpus_path = 'data/{}_corpus.json'.format(self.corpus_name)
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        self.corpus = json.load(open(self.corpus_path, 'r'))
        self.dataset_df = pd.DataFrame()
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]
        self.dataset_df['idx'] = [p['idx'] if 'idx' in p else i for i, p in enumerate(self.corpus)]

    def load_index_files(self):
        index_file_pattern = 'output/openie_{}_results_{}_{}_*.json'.format(self.corpus_name, self.extraction_type, self.extraction_model_name_processed)
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(
                f'No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done.')
            return
        max_samples = np.max(
            [int(file.split('{}_'.format(self.extraction_model_name_processed))[1].split('.json')[0]) for file in
             possible_files])
        extracted_file = json.load(
            open('output/openie_{}_results_{}_{}_{}.json'.format(self.corpus_name, self.extraction_type, self.extraction_model_name_processed, max_samples), 'r'))

        self.extracted_triples = extracted_file['docs']

        if self.corpus_name == 'hotpotqa':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        if self.corpus_name == 'hotpotqa_train':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif 'musique' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif self.corpus_name == '2wikimultihopqa':
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
            self.dataset_df['title'] = [s['title'] for s in self.extracted_triples]
        elif 'case_study' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        else:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]

        if not self.extraction_model_name.startswith('gpt-3.5-turbo'):
            self.extraction_type = self.extraction_type + '_' + self.extraction_model_name_processed
        self.kb_node_phrase_to_id = pickle.load(open(
            'output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type,
                                                                      self.phrase_type,
                                                                      self.extraction_type, self.version),
            'rb'))  # node phrase string -> phrase id
        triple_to_id_path = 'output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version)
        self.triplet_to_id_dict = pickle.load(open(triple_to_id_path, 'rb'))  # fact string -> fact id
        self.triplet_id_to_triple_dict = {v: k for k, v in self.triplet_to_id_dict.items()}

        try:
            node_pair_to_edge_label_path = 'output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(
                self.corpus_name, self.graph_type, self.phrase_type,
                self.extraction_type, self.graph_creating_retriever_name_processed, self.version)
            self.node_pair_to_edge_label_dict = pickle.load(open(node_pair_to_edge_label_path, 'rb'))
        except:
            self.logger.exception('Node pair to edge label dict not found: ' + node_pair_to_edge_label_path)

        self.triples: List = list(self.triplet_to_id_dict.keys())
        self.triples: List = [self.triples[i] for i in np.argsort(list(self.triplet_to_id_dict.values()))]
        self.triples_str_list: List = [str(triple) for triple in self.triples]
        self.node_phrases: np.ndarray = np.array(list(self.kb_node_phrase_to_id.keys()))[np.argsort(list(self.kb_node_phrase_to_id.values()))]

        self.doc_triple_to_freq: Dict = pickle.load(open(
            'output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'rb'))  # doc id, fact id -> frequency (mostly 1)

        self.triple_to_docs: Dict = defaultdict(set)
        for doc_id_triple_id in self.doc_triple_to_freq:
            corpus_doc_id = doc_id_triple_id[0]
            triple_id = doc_id_triple_id[1]
            self.triple_to_docs[triple_id].add(corpus_doc_id)

        self.doc_to_triple: Dict = defaultdict(set)
        for doc_id_triple_id in self.doc_triple_to_freq:
            corpus_doc_id = doc_id_triple_id[0]
            triple_id = doc_id_triple_id[1]
            self.doc_to_triple[corpus_doc_id].add(triple_id)

        self.triples_to_phrases = pickle.load(open(
            'output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'rb'))  # fact id, phrase id -> frequency (mostly 1)

        self.docs_to_triples_mat = pickle.load(open(
            'output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'rb'))  # (num docs, num facts)
        self.triples_to_phrases_mat = pickle.load(open(
            'output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type, self.extraction_type, self.version),
            'rb'))  # (num facts, num phrases)

        self.docs_to_phrases_mat = self.docs_to_triples_mat.dot(self.triples_to_phrases_mat)
        self.docs_to_phrases_mat[self.docs_to_phrases_mat.nonzero()] = 1
        self.phrase_to_num_doc = self.docs_to_phrases_mat.sum(0).T

        graph_file_path = 'output/{}_{}_graph_mean_{}_thresh_{}_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, str(self.sim_threshold), self.phrase_type,
                                                                                          self.extraction_type, self.graph_creating_retriever_name_processed, self.version)
        if os.path.isfile(graph_file_path):
            self.graph_plus = pickle.load(
                open(graph_file_path, 'rb'))  # (phrase1 id, phrase2 id) -> the number of occurrences
        else:
            self.logger.error('Graph file not found: ' + graph_file_path)

        num_triple = self.docs_to_triples_mat.shape[1]
        self.edge_source_nodes = np.zeros(num_triple, dtype=int)
        self.edge_target_nodes = np.zeros(num_triple, dtype=int)
        for t, t_id in self.triplet_to_id_dict.items():
            start_id = self.kb_node_phrase_to_id[t[0]]
            end_id = self.kb_node_phrase_to_id[t[2]]
            self.edge_source_nodes[t_id] = start_id
            self.edge_target_nodes[t_id] = end_id

    def get_phrases_in_doc_by_str(self, doc: str):
        # find doc id from self.dataset_df
        try:
            doc_id = self.dataset_df[self.dataset_df.paragraph.str.contains(doc, regex=False, na=False)].index[0]
            phrase_ids = self.docs_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            return [self.node_phrases[phrase_id] for phrase_id in phrase_ids]
        except Exception as e:
            self.logger.exception('Error in getting phrases in doc: ' + str(e))
            return []

    def get_phrase_in_doc_by_idx(self, doc_idx):
        """
        Using `idx` column in self.dataset_df to find the row and return the corresponding `paragraph` value
        @param doc_idx:
        @return:
        """
        phrase_ids = self.docs_to_phrases_mat[[doc_idx], :].nonzero()[1].tolist()
        return [self.node_phrases[phrase_id] for phrase_id in phrase_ids]

    def build_graph(self):

        edges = set()

        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)

        for edge, weight in tqdm(self.graph_plus.items(), total=len(self.graph_plus), desc='Building Graph'):
            edge1 = edge[0]
            edge2 = edge[1]

            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[(edge[0], edge[1])]

        self.graph_plus = new_graph_plus

        edges = list(edges)

        n_vertices = len(self.kb_node_phrase_to_id)
        self.g = ig.Graph(n_vertices, edges, directed=self.is_directed_graph)

        self.g.es['weight'] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
        self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')

        if self.node_to_doc_strategy == 'map':
            num_phrase = self.triples_to_phrases_mat.shape[1]
            phrase_edges = [e for e in edges if e[0] < num_phrase and e[1] < num_phrase]
            transition_mat = csr_matrix((np.array([self.graph_plus[(v1, v3)] for v1, v3 in phrase_edges]),
                                         ([v1 for v1, v3 in phrase_edges], [v3 for v1, v3 in phrase_edges])), shape=(num_phrase, num_phrase))
            # normalize the matrix
            row_sums = np.array(transition_mat.sum(axis=1)).flatten()
            inv_row_sums = np.reciprocal(row_sums, where=row_sums != 0)  # Avoid division by zero
            inverse_degree_mat = csr_matrix((inv_row_sums, (np.arange(num_phrase), np.arange(num_phrase))),
                                            shape=(num_phrase, num_phrase))
            self.normalized_transition_mat = inverse_degree_mat.dot(transition_mat)

    def load_triple_vectors(self):
        if self.triple_embeddings is not None:
            return
        triple_embeddings_path = (f'data/lm_vectors/{self.linking_retriever_name_processed}_mean/'
                                  f'fact_embeddings_{self.corpus_name}_'
                                  f'{self.extraction_model_name_processed}_{self.graph_creating_retriever_name_processed}.p')
        if os.path.isfile(triple_embeddings_path):
            self.triple_embeddings = pickle.load(open(triple_embeddings_path, 'rb'))
            self.logger.info(
                'Loaded triple embeddings from: ' + triple_embeddings_path + ', shape: ' + str(self.triple_embeddings.shape))
        else:
            self.triple_embeddings = self.embed_model.encode_text(self.triples_str_list, return_cpu=True,
                                                                  return_numpy=True, norm=True)
            pickle.dump(self.triple_embeddings, open(triple_embeddings_path, 'wb'))
            self.logger.info(
                'Saved triple embeddings to: ' + triple_embeddings_path + ', shape: ' + str(self.triple_embeddings.shape))

    def load_node_vectors(self):
        encoded_string_path = 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(
            self.linking_retriever_name_processed)
        if os.path.isfile(encoded_string_path):
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)
        else:  # use another way to load node vectors
            if self.linking_retriever_name == 'colbertv2':
                return
            kb_node_phrase_embeddings_path = (f'data/lm_vectors/{self.linking_retriever_name_processed}_mean/'
                                              f'kb_node_phrase_embeddings_{self.corpus_name}_'
                                              f'{self.extraction_model_name_processed}_{self.graph_creating_retriever_name_processed}.p')
            if os.path.isfile(kb_node_phrase_embeddings_path):
                self.kb_node_phrase_embeddings = pickle.load(open(kb_node_phrase_embeddings_path, 'rb'))
                if len(self.kb_node_phrase_embeddings.shape) == 3:
                    self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
                self.logger.info('Loaded phrase embeddings from: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))
            else:
                self.kb_node_phrase_embeddings = self.embed_model.encode_text(self.node_phrases.tolist(), return_cpu=True, return_numpy=True, norm=True)
                dir = os.path.dirname(kb_node_phrase_embeddings_path)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                pickle.dump(self.kb_node_phrase_embeddings, open(kb_node_phrase_embeddings_path, 'wb'))
                self.logger.info('Saved phrase embeddings to: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))

    def load_node_vectors_from_string_encoding_cache(self, string_file_path):
        self.logger.info('Loading node vectors from: ' + string_file_path)
        kb_vectors = []
        self.strings = open(string_file_path, 'r').readlines()
        for i in range(len(glob('data/lm_vectors/{}_mean/vecs_*'.format(self.linking_retriever_name_processed)))):
            kb_vectors.append(
                torch.Tensor(pickle.load(open('data/lm_vectors/{}_mean/vecs_{}.p'.format(self.linking_retriever_name_processed, i), 'rb'))))
        kb_mat = torch.cat(kb_vectors)  # a matrix of phrase vectors
        self.strings = [s.strip() for s in self.strings]
        self.string_to_id = {string: i for i, string in enumerate(self.strings)}
        kb_mat = kb_mat.T.divide(torch.linalg.norm(kb_mat, dim=1)).T
        kb_mat = kb_mat.to('cuda')
        kb_only_indices = []
        num_non_vector_phrases = 0
        for i in range(len(self.kb_node_phrase_to_id)):
            phrase = self.node_phrases[i]
            if phrase not in self.string_to_id:
                num_non_vector_phrases += 1

            phrase_id = self.string_to_id.get(phrase, 0)
            kb_only_indices.append(phrase_id)
        self.kb_node_phrase_embeddings = kb_mat[kb_only_indices]  # a matrix of phrase vectors
        self.kb_node_phrase_embeddings = self.kb_node_phrase_embeddings.cpu().numpy()
        self.logger.info('{} phrases did not have vectors.'.format(num_non_vector_phrases))

    def load_dpr_doc_embeddings(self):
        if self.doc_embedding_mat is not None:
            return
        cache_filename = 'data/lm_vectors/{}_mean/{}_doc_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
        if os.path.exists(cache_filename):
            self.doc_embedding_mat = pickle.load(open(cache_filename, 'rb'))
            self.logger.info(f'Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}')
        else:
            self.doc_embeddings = []
            self.doc_embedding_mat = self.embed_model.encode_text(self.dataset_df['paragraph'].tolist(), return_cpu=True, return_numpy=True, norm=True)
            if not os.path.isdir('data/lm_vectors/{}_mean/'.format(self.linking_retriever_name_processed)):
                os.makedirs('data/lm_vectors/{}_mean/'.format(self.linking_retriever_name_processed))
            pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
            self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')

    def run_pagerank_igraph_chunk(self, reset_prob_chunk, damping=None):
        """
        Run PageRank on the graph
        @param reset_prob_chunk: reset probability chunk
        @param damping: dampling factor for PPR algorithm
        @return: PageRank probabilities
        """
        pageranked_probabilities = []
        damping = self.damping if damping is None else damping
        for reset_prob in reset_prob_chunk:
            pageranked_probs = self.g.personalized_pagerank(vertices=range(len(self.kb_node_phrase_to_id)), damping=damping, directed=self.is_directed_graph,
                                                            weights=None, reset=reset_prob, implementation='prpack')

            pageranked_probabilities.append(np.array(pageranked_probs))

        return np.array(pageranked_probabilities)

    def get_colbert_max_score(self, query):
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(queries_).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return max_score

    def get_colbert_real_score(self, query, doc):
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

        docs_ = [doc]
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(docs_).float()

        real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return real_score
