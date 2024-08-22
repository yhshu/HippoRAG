import sys

sys.path.append('.')

import argparse
import json
import os
import pickle
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import DatasetDict, Dataset
from gritlm import GritLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, IntervalStrategy
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from src.pangu.retrieval_api import GritLMRetriever


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_custom_dataset(tokenizer):
    cache = pickle.load(open('data/linker_training/sentence_triple_cache.pkl', 'rb'))

    training = []
    validation = []
    test = []

    gritlm_model = GritLM('GritLM/GritLM-7B', torch_dtype='auto')
    retriever_instruction = 'Given a query, retrieve relevant facts that contribute to answering this query.'
    for file_name in os.listdir('data/linker_training/queries'):
        data = json.load(open(f'data/linker_training/queries/{file_name}', 'r'))
        data = data[:100]
        dataset_label = file_name.split('.')[0]
        if 'train' in dataset_label:
            split = 'train'
        elif 'valid' in dataset_label or 'dev' in dataset_label:
            split = 'validation'
        else:
            split = 'test'

        all_triples = []
        for sample in data:
            custom_id = f"{dataset_label}_{sample['id']}"
            if custom_id in cache:
                all_triples.extend(cache[custom_id]['triples'])

        print(f"Loaded {len(all_triples)} facts from {file_name}")

        retriever = GritLMRetriever([json.dumps(triple) for triple in all_triples], model=gritlm_model, instruction=retriever_instruction)
        for sample in tqdm(data):
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
                {'role': 'system', 'content': generative_reranking_prompt},
                {'role': 'user', 'content': f'\nQuery: {query}\nCandidate facts:\n' + '\n'.join([json.dumps(triple).lower() for triple in retrieved])},
                {'role': 'assistant', 'content': json.dumps({"fact": labels}).lower()}
            ]
            if split == 'train':
                training.append({'text': tokenizer.apply_chat_template(messages, tokenize=False)})
            elif split == 'validation':
                validation.append({'text': tokenizer.apply_chat_template(messages, tokenize=False)})
            else:
                test.append({'text': tokenizer.apply_chat_template(messages, tokenize=False)})

    # end for each dataset file

    # Create Dataset objects
    datasets = {}
    if len(training) > 0:
        datasets['train'] = Dataset.from_list(training)
    if len(validation) > 0:
        datasets['validation'] = Dataset.from_list(validation)
    if len(test) > 0:
        datasets['test'] = Dataset.from_list(test)
    return datasets


def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    print(f"Rank {rank} started, world size: {world_size}")
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

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    model = get_peft_model(model, peft_config)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        model.module.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        model.module.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sampler = DistributedSampler(dataset["train"], num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(dataset["validation"], num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator
    )

    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler,
        collate_fn=data_collator
    )

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=1024,
        output_dir="exp/fact_linker",
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        local_rank=rank
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    if rank == 0:
        trainer.save_model("exp/fact_linker/model")
        tokenizer.save_pretrained("exp/fact_linker/tokenizer")

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--local-rank', type=int)
    args = parser.parse_args()

    local_rank = args.local_rank
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
