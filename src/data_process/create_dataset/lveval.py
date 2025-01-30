import argparse
import copy
import json
import re
from tqdm import tqdm


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compare_texts(normalized_text1, normalized_text2):
    return normalized_text1 == normalized_text2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/LVEval/hotpotwikiqa_mixup/hotpotwikiqa_mixup_256k.jsonl')
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f'#query: {len(data)}')

    corpus = []
    dataset = []
    normalized_corpus_set = set()

    for sample_idx, sample in tqdm(enumerate(data), desc='Collecting data', total=len(data)):
        query = sample['input']
        context = sample['context']

        passage_split = context.split('### Passage ')
        passages = []
        for p in passage_split:
            p = p.strip()
            if not p:
                continue
            paragraphs = '\n'.join(p.split('\n')[1:])  # skip the serial number in the first line
            paragraph_split = paragraphs.split('\n\n')
            for paragraph in paragraph_split:
                paragraph = paragraph.strip()
                if not paragraph or len(paragraph) < 10:
                    continue
                passages.append(paragraph)

        for passage in passages:
            normalized_passage = normalize_text(passage)
            if normalized_passage not in normalized_corpus_set:
                normalized_corpus_set.add(normalized_passage)
                corpus.append({'idx': len(corpus), 'title': '', 'text': passage})

        print(f'#sample {sample_idx}: {len(passages)} passages found, {len(corpus)} passages in total')

        new_sample = copy.deepcopy(sample)
        new_sample['question'] = query
        del new_sample['input']
        new_sample['id'] = sample_idx
        dataset.append(new_sample)
    # end for each sample

    dataset_output_path = 'data/lveval.json'
    with open(dataset_output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f'Dataset {len(dataset)} written to {dataset_output_path}')

    corpus_output_path = 'data/lveval_corpus.json'
    with open(corpus_output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    print(f'Corpus {len(corpus)} written to {corpus_output_path}')
