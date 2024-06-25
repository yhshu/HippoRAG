import argparse
import json
import pickle

import numpy as np
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def colbertv2_index(corpus: list, dataset_name: str, exp_name: str, index_name='nbits_2', checkpoint_path='exp/colbertv2.0', overwrite='reuse'):
    """
    Indexing corpus and phrases using colbertv2
    @param corpus:
    @return:
    """
    corpus_processed = [x.replace('\n', '\t') for x in corpus]

    corpus_tsv_file_path = f'data/lm_vectors/colbert/{dataset_name}_{exp_name}_{len(corpus_processed)}.tsv'
    with open(corpus_tsv_file_path, 'w') as f:  # save to tsv
        for pid, p in enumerate(corpus_processed):
            f.write(f"{pid}\t\"{p}\"" + '\n')
    root_path = f'data/lm_vectors/colbert/{dataset_name}'

    # indexing corpus
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=root_path)):
        config = ColBERTConfig(
            nbits=2,
            root=root_path,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=corpus_tsv_file_path, overwrite=overwrite)


def colbertv2_graph_indexing(dataset_name: str, corpus_path: str, phrase_path: str, checkpoint_path: str = 'exp/colbertv2.0'):
    corpus_path = json.load(open(corpus_path, 'r'))
    # get corpus tsv
    if 'hotpotqa' in dataset_name:
        corpus_contents = [x[0] + ' ' + ''.join(x[1]) for x in corpus_path.items()]
    else:
        corpus_contents = [x['title'] + ' ' + x['text'].replace('\n', ' ') for x in corpus_path]
    colbertv2_index(corpus_contents, dataset_name, 'corpus', checkpoint_path, overwrite=True)
    kb_phrase_dict = pickle.load(open(phrase_path, 'rb'))
    phrases = np.array(list(kb_phrase_dict.keys()))[np.argsort(list(kb_phrase_dict.values()))]
    phrases = phrases.tolist()
    colbertv2_index(phrases, dataset_name, 'phrase', checkpoint_path, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--phrase', type=str)
    args = parser.parse_args()

    colbertv2_graph_indexing(args.dataset, args.corpus, args.phrase)
