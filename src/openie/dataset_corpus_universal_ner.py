import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def universal_ner_type_prompt(input_text, entity_type):
    prompt = f"""A virtual assistant answers questions from a user based on the provided text.
USER: Text: {input_text}
ASSISTANT: I’ve read this text.
USER: What describes {entity_type} in the text?
ASSISTANT:"""
    return prompt


entity_type_list = ['person', 'organization', 'location', 'product', 'event']


def ner(text, entity_type, model, tokenizer, device):
    prompt = universal_ner_type_prompt(text, entity_type)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=96)
    ner_results = tokenizer.decode(output[0], skip_special_tokens=True)
    ner_results = ner_results[len(prompt):]
    try:
        ner_results = json.loads(ner_results)
        if not isinstance(ner_results, list):
            print(f"Failed to extract {entity_type} from: {text}; NER: ", ner_results)
            ner_results = []
    except Exception as e:
        print(f"Failed to extract {entity_type} from: {text}; NER: ", ner_results)
        ner_results = []
    return ner_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='Universal-NER/UniNER-7B-all')  # https://huggingface.co/Universal-NER
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    print(f"Using {torch.cuda.device_count()} devices")

    dataset = json.load(open(f'data/{args.dataset}.json'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model_label = args.model.split('/')[-1]

    all_entities = {t: set() for t in entity_type_list}
    for passage in tqdm(corpus, total=len(corpus), desc='Extracting entities from corpus'):
        passage_str = passage['text']
        for t in entity_type_list:
            ner_results = ner(passage_str, t, model, tokenizer, args.device)
            passage[f'universal_ner_{t}'] = ner_results
            all_entities[t].update(ner_results)

    for sample in tqdm(dataset, desc='Extracting entities from dataset'):
        query = sample['question']
        for t in entity_type_list:
            ner_results = ner(query, t, model, tokenizer, args.device)
            sample[f'universal_ner_{t}'] = ner_results
            all_entities[t].update(ner_results)

        for d in sample['question_decomposition']:
            for t in entity_type_list:
                ner_results = ner(d['answer'], t, model, tokenizer, args.device)
                sample[f'universal_ner_{t}'] = ner_results
                all_entities[t].update(ner_results)

    # save results
    ner_output_path = f'data/universal_ner_{args.dataset}_{model_label}_entities.json'
    for t in entity_type_list:
        all_entities[t] = list(all_entities[t])
    with open(ner_output_path, 'w') as f:
        json.dump(all_entities, f)
    print(f"Saved to {ner_output_path}, len: {len(all_entities)}")

    corpus_output_path = f'data/universal_ner_{args.dataset}_{model_label}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(corpus, f)
    print(f"Saved to {corpus_output_path}, len: {len(corpus)}")

    dataset_output_path = f'data/universal_ner_{args.dataset}_{model_label}.json'
    with open(dataset_output_path, 'w') as f:
        json.dump(dataset, f)
    print(f"Saved to {dataset_output_path}, len: {len(dataset)}")
