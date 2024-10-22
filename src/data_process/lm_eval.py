import argparse
import json

from tqdm import tqdm
from nltk import sent_tokenize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/LVEval/hotpotwikiqa_mixup/hotpotwikiqa_mixup_256k.jsonl')
    args = parser.parse_args()

    # read jsonl from LM Eval
    with open(args.data, 'r') as f:
        data = [json.loads(line) for line in f]
    print('#query', len(data))

    corpus = []
    dataset = []

    for sample in tqdm(data):
        query = sample['input']
        context = sample['context']

        sentences = sent_tokenize(context)
        # split sentences into passages with <=5 sentences
        passages = []
        passage = []
        for sent in sentences:
            passage.append(sent)
            if len(passage) == 5:
                passages.append(' '.join(passage))
                passage = []
        if passage:
            passages.append(' '.join(passage))

        for passage in passages:
            corpus.append({'idx': len(corpus), 'text': passage})
