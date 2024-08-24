import sys

sys.path.append('.')

from tqdm import tqdm
import argparse
import json
import os.path

import torch
from peft import PeftConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.linking.llama3_fact_linker_lora_train import load_custom_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='lora dir', default='exp/fact_linker')
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    if args.model is not None:
        if args.ckpt is None:
            peft_config = PeftConfig.from_pretrained(os.path.join(args.model, 'model'))
        else:
            peft_config = PeftConfig.from_pretrained(os.path.join(args.model, args.ckpt))

        base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', device_map='auto', return_dict=True)
        model = PeftModel.from_pretrained(base_model, os.path.join(args.model, 'model'))
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model, 'tokenizer'))
    else:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    datasets = load_custom_dataset()
    metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    for idx, sample in tqdm(enumerate(datasets['validation']), total=len(datasets['validation']), desc='Evaluating'):
        messages = json.loads(sample['text'])
        try:
            gold_completion = eval(messages[2]['content']).get('fact', [])
            gold_completion = [tuple(t) for t in gold_completion]
        except Exception as e:
            print(e)
            continue
        with torch.no_grad():
            input_text = tokenizer.apply_chat_template(messages[:2], tokenize=False)
            inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
            input_length = inputs['input_ids'].shape[-1]

            outputs = model.generate(**inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id)
            completion = outputs[:, input_length:]

            output_text = tokenizer.decode(completion[0])
            output_text = output_text.split('<|end_header_id|>')[1].split('<|eot_id|>')[0].strip()

        try:
            completion = json.loads(output_text).get('fact', [])
            completion = [tuple(t) for t in completion]
        except Exception as e:
            print(e)
            completion = []

        # Calculate precision, recall, and F1 between the gold completion and the generated completion
        precision = len(set(gold_completion) & set(completion)) / len(set(completion)) if len(set(completion)) > 0 else 0
        recall = len(set(gold_completion) & set(completion)) / len(set(gold_completion)) if len(set(gold_completion)) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics['precision'] += precision
        metrics['recall'] += recall
        metrics['f1'] += f1

        print('[GOLD]', gold_completion)
        print('[PREDICTION]', completion)
        for k, v in metrics.items():
            print(f'{k}: {round(v / (idx + 1), 4)}')
        print()

    for k, v in metrics.items():
        print(f'{k}: {round(v / len(datasets["validation"]), 4)}')
