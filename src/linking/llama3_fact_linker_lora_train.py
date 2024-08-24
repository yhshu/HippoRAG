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


def load_custom_dataset(tokenizer=None, data_dir='data/linker_training/queries'):
    cache = pickle.load(open('data/linker_training/sentence_triple_cache.pkl', 'rb'))
    gritlm_model = GritLM('GritLM/GritLM-7B', torch_dtype='auto')
    datasets = {}

    for file_name in os.listdir(data_dir):
        dataset_label = file_name.split('.')[0]
        if 'train' in dataset_label:
            split = 'train'
        elif 'valid' in dataset_label or 'dev' in dataset_label:
            split = 'validation'
        else:
            split = 'test'
        split_data = load_dataset_split(cache, data_dir, file_name, gritlm_model, tokenizer, dataset_label)
        if len(split_data) > 0:
            datasets[split] = Dataset.from_list(split_data)
    # end for each dataset file

    dataset_dict = DatasetDict(datasets)
    return dataset_dict


def load_dataset_split(cache, data_dir: str, file_name: str, gritlm_model, tokenizer, dataset_label: str):
    os.makedirs('data/linker_training/samples', exist_ok=True)
    tokenization_str = 'tokenized' if tokenizer is not None else 'chatml'
    linking_data_file_name = 'data/linker_training/samples/' + dataset_label + f'_{tokenization_str}.json'
    if os.path.exists(linking_data_file_name):
        return json.load(open(linking_data_file_name, 'r'))

    res = []
    input_data = json.load(open(os.path.join(data_dir, file_name), 'r'))
    all_triples = []
    for sample in input_data:
        custom_id = f"{dataset_label}_{sample['id']}"
        if custom_id in cache:
            all_triples.extend(cache[custom_id]['triples'])
    print(f"Loaded {len(all_triples)} facts from {file_name}")
    retriever = GritLMRetriever([json.dumps(triple) for triple in all_triples], model=gritlm_model, instruction=fact_retriever_instruction)

    for sample in tqdm(input_data):
        query = sample['question']
        labels = []  # facts to link to

        custom_id = f"{dataset_label}_{sample['id']}"
        if custom_id in cache:
            labels = cache[custom_id]['triples']

        retrieved = retriever.get_top_k_sentences(query, k=30)
        retrieved = [eval(triple) for triple in retrieved]
        # check if labels are in retrieved facts
        for label in labels:
            if label not in retrieved:
                retrieved = [label] + retrieved[:-1]
        # order retrieved facts by subject
        retrieved = sorted(retrieved, key=lambda x: x[0])

        from src.rerank import generative_reranking_prompt
        messages = [
            SystemMessage(generative_reranking_prompt),
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
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    dataset = load_custom_dataset(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
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
        output_dir="exp/fact_linker",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
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
    trainer.save_model("exp/fact_linker/model")
    tokenizer.save_pretrained("exp/fact_linker/tokenizer")
