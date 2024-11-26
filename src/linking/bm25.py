import numpy as np
from src.hipporag import HippoRAG


def bm25_retrieval(hipporag: HippoRAG, query: str, logs=None):
    indices, scores = hipporag.bm25_retriever.get_top_k_indices(query, 10, True, True)
    return np.array(indices), np.array(scores), logs
