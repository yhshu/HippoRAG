import argparse
import json
import os
import pickle

from openai import OpenAI
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_id', type=str)
    args = parser.parse_args()

    client = OpenAI(max_retries=5, timeout=60)
    content = client.files.content(file_id=args.file_id)

    # save file to cache
    lines = content.text.strip().split("\n")
    print('content length:', len(lines))

    cache_path = 'data/linker_training/sentence_triple_cache.pkl'
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, 'rb'))
    else:
        cache = {}  # custom_id -> {sentences, triples}

    print('Batch output length:', len(lines))
    for line in tqdm(lines):
        response = json.loads(line)
        try:
            content = json.loads(response['response']['body']['choices'][0]['message']['content'])
        except:
            print('Error:', response)
            continue
        custom_id = response['custom_id']
        cache[custom_id] = content  # write to cache: extracted triples

    pickle.dump(cache, open(cache_path, 'wb'))
    print('Cache saved to', cache_path)
