# https://llama.meta.com/docs/how-to-guides/fine-tuning/
# https://github.com/huggingface/peft
import sys

sys.path.append('.')

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.util.llama_cpp_service import langchain_message_to_llama_3_prompt, langchain_message_to_chatml

from transformers.trainer_utils import IntervalStrategy
import argparse
import json
import os
import pickle

from datasets import DatasetDict, Dataset
from gritlm import GritLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from src.pangu.retrieval_api import GritLMRetriever

fact_retriever_instruction = 'Given a query, retrieve relevant facts that contribute to answering this query.'


def load_custom_dataset(tokenizer=None, data_dir='data/linker_training/queries', selected_datasets=None, num_candidate_fact=30):
    cache = pickle.load(open('data/linker_training/sentence_triple_cache.pkl', 'rb'))
    gritlm_model = GritLM('GritLM/GritLM-7B', torch_dtype='auto')
    datasets = {}

    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.json') and not file_name.endswith('.jsonl'):
            continue
        dataset_label = file_name.split('.')[0]
        if selected_datasets is not None and not any(dataset in dataset_label for dataset in selected_datasets):
            continue

        if 'train' in dataset_label:
            split = 'train'
        elif 'valid' in dataset_label or 'dev' in dataset_label:
            split = 'validation'
        else:
            split = 'test'
        split_data = load_dataset_split(cache, data_dir, file_name, gritlm_model, tokenizer, dataset_label, num_candidate_fact)
        if len(split_data) > 0:
            datasets[split] = Dataset.from_list(split_data)
    # end for each dataset file

    dataset_dict = DatasetDict(datasets)
    return dataset_dict


def retrieved_items_to_candidate_facts(retrieved, labels, k=30):
    retrieved = [eval(triple) for triple in retrieved]
    labels_dict = {(t[0], t[2]): t for t in labels if len(t) == 3}

    # remove triples with duplicate subjects and objects
    retrieved_dict = {}
    for t in retrieved:
        if len(t) != 3:
            continue
        if (t[0], t[2]) not in retrieved_dict and (t[0], t[2]) not in labels_dict:
            retrieved_dict[(t[0], t[2])] = t
    retrieved = list(retrieved_dict.values())

    # check if labels are in retrieved facts
    for label in labels:
        if label not in retrieved and len(label) == 3:
            retrieved = [label] + retrieved[:-1]

    # order retrieved facts by subject
    retrieved = sorted(retrieved, key=lambda x: x[0])
    res = retrieved[:k]
    return res


def load_triples(cache: dict, dataset_label: str, sample: dict):
    from src.data_process.util import generate_hash
    if 'msmacro' in dataset_label:
        for candidate in sample['candidates']:
            custom_id = f"{dataset_label}_{sample['id']}_{generate_hash(candidate['sentence'])}"
            if custom_id in cache:
                return cache[custom_id]['triples']
    else:
        custom_id = f"{dataset_label}_{sample['id']}"
        if custom_id in cache:
            return cache[custom_id]['triples']
    return []


def load_dataset_split(cache, data_dir: str, file_name: str, gritlm_model, tokenizer, dataset_label: str, num_candidate_fact=30):
    os.makedirs('data/linker_training/samples', exist_ok=True)
    tokenization_str = 'tokenized' if tokenizer is not None else 'chatml'
    linking_data_file_name = 'data/linker_training/samples/' + dataset_label + f'_{tokenization_str}.json'
    if os.path.exists(linking_data_file_name):
        return json.load(open(linking_data_file_name, 'r'))

    res = []
    assert os.path.exists(os.path.join(data_dir, file_name)), f"File {file_name} not found in {data_dir}"
    try:
        input_data = json.load(open(os.path.join(data_dir, file_name), 'r'))
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        exit(1)
    all_triples = []
    for sample in input_data:
        all_triples.extend(load_triples(cache, dataset_label, sample))
    print(f"Loaded {len(all_triples)} facts from {file_name}")
    retriever = GritLMRetriever([json.dumps(triple) for triple in all_triples], model=gritlm_model, instruction=fact_retriever_instruction)

    for sample in tqdm(input_data):
        query = sample['question']
        labels = []  # facts to link to

        custom_id = f"{dataset_label}_{sample['id']}"
        if custom_id in cache:
            labels = cache[custom_id]['triples']

        retrieved = retriever.get_top_k_sentences(query, k=num_candidate_fact)
        retrieved = retrieved_items_to_candidate_facts(retrieved, labels, k=num_candidate_fact)

        from src.rerank.prompt import generative_multi_hop_filter_prompt
        messages = [
            SystemMessage(generative_multi_hop_filter_prompt),
            HumanMessage(f'\nQuery: {query}\nCandidate facts:\n' + '\n'.join([json.dumps(triple).lower() for triple in retrieved])),
            AIMessage(json.dumps({'fact': labels}).lower())
        ]
        if tokenizer is not None:
            res.append({'text': langchain_message_to_llama_3_prompt(messages, False)})
        else:
            res.append({'text': json.dumps(langchain_message_to_chatml(messages))})

    # save the dataset to linking_data_file_name
    with open(linking_data_file_name, 'w') as f:
        json.dump(res, f)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', type=str, help='A list of datasets')
    parser.add_argument('--exp', type=str, help='Experiment name', default='fact_linker')
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('-nc', '--num_candidate_fact', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(f"exp/{args.exp}", exist_ok=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_custom_dataset(tokenizer, selected_datasets=args.datasets, num_candidate_fact=args.num_candidate_fact)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    if args.use_peft:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        model.resize_token_embeddings(len(tokenizer))


    def preprocess_function(examples):
        inputs = tokenizer(examples['text'],
                           padding="max_length",
                           truncation=True,
                           max_length=1024)
        return inputs


    # tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Data collator
    data_collator = DataCollatorForCompletionOnlyLM(response_template='<|start_header_id|>assistant<|end_header_id|>',
                                                    tokenizer=tokenizer)

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=1024,
        output_dir=f"exp/{args.exp}",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=5000,
        logging_strategy=IntervalStrategy.EPOCH,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(f"exp/{args.exp}/adapter")
