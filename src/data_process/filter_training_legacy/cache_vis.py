import os
import pickle

if __name__ == '__main__':

    cache_path = 'data/linker_training/sentence_triple_cache.pkl'
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, 'rb'))
        print('len cache', len(cache))
