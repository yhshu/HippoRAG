import sys

sys.path.append('.')
from src.lm_wrapper.gritlm import GritLMWrapper
from src.lm_wrapper.sentence_transformers_util import SentenceTransformersWrapper

import _pickle as pickle
import argparse
import os.path

import numpy as np
import pandas as pd

import pickle
import os

import torch
from tqdm import tqdm

import gc

from transformers import AutoModel, AutoTokenizer

from src.processing import processing_phrases, mean_pooling

# TODO: Change hard-coded vector output directory
VECTOR_DIR = 'data/lm_vectors'


class RetrievalModule:
    """
    Class designed to retrieve potential synonymy candidates for a set of UMLS terms from a set of entities.
    """

    def __init__(self,
                 retriever_name,
                 string_filename,
                 dataset_name,
                 pool_method='cls',
                 ):
        """
        Args:
            retriever_name: Retrieval names can be one of 3 types
                2) The name of a pickle file mapping AUIs to precomputed vectors
                3) A huggingface transformer model
        """

        self.retriever_name = retriever_name
        self.retrieval_name_dir = None
        self.pool_method = pool_method
        assert dataset_name is not None
        self.dataset_name = dataset_name.replace('/', '_').replace('.', '_')

        # Search for pickle file
        print('No Pre-Computed Vectors. Confirming PLM Model.')

        try:
            if 'GritLM' in retriever_name:
                self.plm = GritLMWrapper(retriever_name)
                self.encode_strings_func = self.encode_strings_wrapper
            elif 'Qwen' in retriever_name:
                self.plm = SentenceTransformersWrapper(retriever_name)
                self.encode_strings_func = self.encode_strings_wrapper
            elif 'nvidia/NV-Embed' in retriever_name:
                from src.lm_wrapper.nv_embed import NVEmbedV2Wrapper
                self.plm = NVEmbedV2Wrapper(retriever_name)
                self.encode_strings_func = self.encode_strings_wrapper
            else:
                if 'ckpt' in retriever_name:
                    self.plm = AutoModel.load_from_checkpoint(retriever_name)
                else:
                    self.plm = AutoModel.from_pretrained(retriever_name)
                self.encode_strings_func = self.encode_strings
        except Exception as e:
            print(e)
            print('Loading {} failed. Possible reasons include: 1. Please make sure it is a valid model name; 2. GPU memory.'.format(retriever_name))
            assert False

        # If not pre-computed, create vectors
        self.retrieval_name_dir = VECTOR_DIR + '/' + self.retriever_name.replace('/', '_').replace('.', '') + '_' + pool_method

        if not (os.path.exists(self.retrieval_name_dir)):
            os.makedirs(self.retrieval_name_dir)
            print('Creating Directory: {}'.format(self.retrieval_name_dir))

        # Get previously computed vectors
        precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors(self.retrieval_name_dir)
        print('len precomp_strings: ', len(precomp_strings))
        print('len precomp_vectors: ', len(precomp_vectors))

        # Get AUI Strings to be Encoded
        string_df = pd.read_csv(string_filename, sep='\t')
        string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
        sorted_df = self.create_sorted_df(string_df.strings.values)

        # Identify Missing Strings
        missing_strings = self.find_missing_strings(sorted_df.strings.unique(), precomp_strings)

        # Encode Missing Strings
        if len(missing_strings):
            print('Encoding {} Missing Strings'.format(len(missing_strings)))
            new_vectors, new_strings, = self.encode_strings_func(missing_strings)

            precomp_strings = list(precomp_strings)
            precomp_vectors = list(precomp_vectors)

            precomp_strings.extend(list(new_strings))
            precomp_vectors.extend(list(new_vectors))

            precomp_vectors = np.array(precomp_vectors)

            self.save_vectors(precomp_strings, precomp_vectors, self.retrieval_name_dir)

        self.vector_dict = self.make_dictionary(sorted_df, precomp_strings, precomp_vectors)

        print('Vectors Loaded.')

        queries = string_df[string_df.type == 'query']
        kb = string_df[string_df.type == 'kb']

        nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values)
        pickle.dump(nearest_neighbors, open(self.retrieval_name_dir + '/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0]), 'wb'))

    def get_precomputed_plm_vectors(self, retrieval_name_dir):

        # Load or Create a DataFrame sorted by phrase length for efficient PLM computation
        strings = self.load_precomp_strings(retrieval_name_dir)
        vectors = self.load_plm_vectors(retrieval_name_dir)

        return strings, vectors

    def create_sorted_df(self, strings):
        lengths = []

        for string in tqdm(strings):
            lengths.append(len(str(string)))

        lengths_df = pd.DataFrame(lengths)
        lengths_df['strings'] = strings

        return lengths_df.sort_values(0)

    def load_precomp_strings(self, retrieval_name_dir):
        filename = retrieval_name_dir + f'/encoded_strings_{self.dataset_name}.txt'

        if not (os.path.exists(filename)):  # No precomputed data
            return []

        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

        return lines

    def load_plm_vectors(self, retrieval_name_dir):
        vectors = []

        print('Loading PLM Vectors.')
        file_path = retrieval_name_dir + f"/vecs_{self.dataset_name}.p"
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                vectors = pickle.load(f)
                print('Loaded {} vectors from {}'.format(len(vectors), file_path))
        return vectors

    def save_vectors(self, strings, vectors, dir_path):
        with open(dir_path + f'/encoded_strings_{self.dataset_name}.txt', 'w') as f:
            for string in strings:
                f.write(string + '\n')

        file_path = dir_path + f'/vecs_{self.dataset_name}.p'
        print('Saving {} vectors to {}'.format(len(vectors), file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(vectors, f)
        print('Saved {} vectors to {}'.format(len(vectors), file_path))

    def find_missing_strings(self, relevant_strings, precomputed_strings):

        return list(set(relevant_strings).difference(set(precomputed_strings)))

    def make_dictionary(self, sorted_df, precomp_strings, precomp_vectors):

        print('Populating Vector Dict')
        precomp_string_ids = {}

        for i, string in enumerate(precomp_strings):
            precomp_string_ids[string] = i

        vector_dict = {}

        for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
            string = row.strings

            try:
                vector_id = precomp_string_ids[string]
                vector_dict[string] = precomp_vectors[vector_id]
            except Exception as e:
                print('make dictionary exception: {}'.format(e))

        return vector_dict

    def encode_strings_wrapper(self, strs_to_encode):
        all_embeddings = self.plm.encode_text(strs_to_encode, return_numpy=True, return_cpu=True, norm=True)
        return all_embeddings, strs_to_encode

    def encode_strings(self, strs_to_encode):
        self.plm.to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)

        # Sorting Strings by length
        sorted_missing_strings = [len(s) for s in strs_to_encode]
        strs_to_encode = list(np.array(strs_to_encode)[np.argsort(sorted_missing_strings)])

        all_cls = []
        all_strings = []
        num_strings_proc = 0

        with torch.no_grad():

            batch_sizes = []

            text_batch = []
            max_pad_size = 0

            for i, string in tqdm(enumerate(strs_to_encode), total=len(strs_to_encode)):

                length = len(tokenizer.tokenize(string))

                text_batch.append(string)
                num_strings_proc += 1

                if length > max_pad_size:
                    max_pad_size = length

                if max_pad_size * len(text_batch) > 50000 or num_strings_proc == len(strs_to_encode):

                    text_batch = list(text_batch)
                    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.plm.config.max_length)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    input_ids = input_ids.to('cuda')
                    attention_mask = attention_mask.to('cuda')

                    outputs = self.plm(input_ids, attention_mask=attention_mask)

                    if self.pool_method == 'cls':
                        embeddings = outputs[0][:, 0, :]

                    elif self.pool_method == 'mean':
                        embeddings = mean_pooling(outputs[0], attention_mask)

                    all_cls.append(embeddings.cpu().numpy())
                    all_strings.extend(text_batch)

                    batch_sizes.append(len(text_batch))

                    text_batch = []
                    max_pad_size = 0

        all_cls = np.vstack(all_cls)

        assert len(all_cls) == len(all_strings)
        assert all([all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))])

        return all_cls, all_strings


    def retrieve_knn(self, queries, knowledge_base, k=2047, batch_size=1000, index_batch_size=10000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        original_vecs = [self.vector_dict[s] for s in knowledge_base]

        if len(original_vecs) == 0:
            return {}

        original_vecs = torch.tensor(original_vecs, dtype=torch.float32)
        original_vecs = torch.nn.functional.normalize(original_vecs, dim=1)

        new_vecs = [self.vector_dict[s] for s in queries]
        if len(new_vecs) == 0:
            return {}

        new_vecs = torch.tensor(new_vecs, dtype=torch.float32)
        new_vecs = torch.nn.functional.normalize(new_vecs, dim=1)

        result = {}

        def get_query_batches(query_vecs, batch_size):
            for i in range(0, len(query_vecs), batch_size):
                yield query_vecs[i:i + batch_size], i

        def get_knowledge_base_batches(knowledge_base_vecs, index_batch_size):
            for i in range(0, len(knowledge_base_vecs), index_batch_size):
                yield knowledge_base_vecs[i:i + index_batch_size], i

        for query_batch, batch_start_idx in get_query_batches(new_vecs, batch_size):
            query_batch = query_batch.clone().detach()
            query_batch = query_batch.to(device)

            batch_similarities = []
            batch_indices = []

            offset_kb = 0

            for kb_batch, kb_start_idx in get_knowledge_base_batches(original_vecs, index_batch_size):
                kb_batch = kb_batch.to(device)
                batch_size_kb = kb_batch.size(0)

                similarity = torch.mm(query_batch, kb_batch.T)

                similarities, indices = torch.topk(similarity, min(k, batch_size_kb), dim=1, largest=True, sorted=True)

                indices += offset_kb

                batch_similarities.append(similarities)
                batch_indices.append(indices)

                del similarity
                kb_batch = kb_batch.cpu()
                torch.cuda.empty_cache()

                offset_kb += batch_size_kb
            # end for each kb batch

            batch_similarities = torch.cat(batch_similarities, dim=1)
            batch_indices = torch.cat(batch_indices, dim=1)

            final_similarities, final_indices = torch.topk(batch_similarities, min(k, batch_similarities.size(1)), dim=1, largest=True, sorted=True)
            final_indices = final_indices.cpu()
            final_similarities = final_similarities.cpu()

            for i in range(final_indices.size(0)):
                query_idx = batch_start_idx + i
                query = queries[query_idx]
                indices_i = final_indices[i]
                similarities_i = final_similarities[i]

                global_indices = batch_indices[i][indices_i]

                knn_strings = [knowledge_base[idx] for idx in global_indices.cpu().numpy()]
                result[query] = (knn_strings, similarities_i.numpy().tolist())

            query_batch = query_batch.cpu()
            torch.cuda.empty_cache()
        # end for each query batch

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_name', type=str, help='retrieval model name, e.g., "facebook/contriever"')
    parser.add_argument('--string_filename', type=str)
    parser.add_argument('--pool_method', type=str, default='mean')
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    retriever_name = args.retriever_name
    string_filename = args.string_filename
    pool_method = args.pool_method

    retrieval_module = RetrievalModule(retriever_name, string_filename, args.dataset_name, pool_method)
