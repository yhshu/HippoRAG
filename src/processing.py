import json
import re

import numpy as np
import torch

from src.data_process.util import generate_hash


def get_file_name(path):
    return path.split('/')[-1].replace('.jsonl', '').replace('.json', '')


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def mean_pooling_embedding(input_str: str, tokenizer, model, device='cuda'):
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)

    embedding = mean_pooling(outputs[0], inputs['attention_mask']).to('cpu').detach().numpy()
    return embedding


def mean_pooling_embedding_with_normalization(input_str, tokenizer, model, device='cuda'):
    encoding = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs[0], attention_mask)
    embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T

    return embeddings


def processing_phrases(phrase):
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()


def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ''
    else:
        return ''


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def retain_top_n_elements(arr: np.ndarray, n=10):
    indices = np.argsort(arr)[-n:]

    new_arr = np.zeros_like(arr)
    new_arr[indices] = arr[indices]

    return new_arr


def z_score_filtering(arr: np.ndarray, threshold=-0.5):
    """
    Calculate the z-score of each **non-zero** element in the array, and filter out the elements with z-score less than the threshold
    @param arr:
    @param threshold:
    @return:
    """
    mask = (arr != 0)
    z_scores = np.zeros_like(arr)
    z_scores[mask] = (arr[mask] - np.mean(arr[mask])) / np.std(arr[mask])

    new_arr = np.zeros_like(arr)
    new_arr[mask] = arr[mask] * (z_scores[mask] > threshold)

    return new_arr


def score_threshold_filtering(arr: np.ndarray, threshold=0.1):
    """
    Filter out the elements with score less than the threshold
    @param arr:
    @param threshold:
    @return:
    """
    mask = (arr > threshold)
    new_arr = np.zeros_like(arr)
    new_arr[mask] = arr[mask]

    if np.sum(mask) == 0:
        return arr
    return new_arr


import numpy as np
from scipy.stats import entropy


def entropy_based_truncation(arr: np.ndarray):
    """
    Truncate the sparse array based on the entropy of the non-zero elements
    @param arr:
    @return:
    """
    non_zero_indices = arr != 0
    non_zero_elements = arr[non_zero_indices]

    if len(non_zero_elements) <= 1:
        return arr

    non_zero_sum = np.sum(non_zero_elements)
    normalized_elements = non_zero_elements / non_zero_sum

    score_entropy = entropy(normalized_elements)

    truncation_factor = max(0.5, score_entropy / np.log(len(non_zero_elements)))

    num_to_keep = int(truncation_factor * len(non_zero_elements))

    sorted_indices = np.argsort(non_zero_elements)[::-1]
    keep_indices = sorted_indices[:num_to_keep]

    truncated_array = np.copy(arr)

    drop_indices = sorted_indices[num_to_keep:]
    truncated_array[non_zero_indices][drop_indices] = 0

    return truncated_array


def softmax_with_zeros(logits):
    mask = (logits != 0)

    exp_logits = np.exp(logits[mask] - np.max(logits[mask]))
    probabilities = np.zeros_like(logits)
    probabilities[mask] = exp_logits / np.sum(exp_logits)

    return probabilities


def deduplicate_triples(triples: list):
    unique_triples = set()
    deduplicated_triples = []
    for triple in triples:
        if tuple(triple) not in unique_triples:
            unique_triples.add(tuple(triple))
            deduplicated_triples.append(triple)

    return deduplicated_triples


def fix_broken_generated_json(json_str: str):
    last_comma_index = json_str.rfind(',')
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    processed_string = json_str + ']\n}'
    return processed_string


def eval_json_str(json_str):
    try:
        return eval(json_str)
    except Exception as e1:
        try:
            return eval(fix_broken_generated_json(json_str))
        except Exception as e2:
            return ''

def check_corpus_duplication(corpus: list):
    hash_set = set()
    for item in corpus:
        content = item['title'] + '\n' + item['text']
        content_hash = generate_hash(content)
        if content_hash not in hash_set:
            hash_set.add(content_hash)
            continue
        else:
            print('Duplicated content hash:', content_hash)
            return False
    return True