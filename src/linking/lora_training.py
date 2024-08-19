# https://llama.meta.com/docs/how-to-guides/fine-tuning/
# https://github.com/huggingface/peft

import sys

sys.path.append('.')
import json
import os
import pickle

from datasets import DatasetDict, Dataset
from gritlm import GritLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from src.pangu.retrieval_api import GritLMRetriever

generative_reranking_prompt = """You are an expert in ranking facts based on their relevance to the query. 

- Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
- If the query is a claim, relevance means the fact supports or contradicts it. For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
- Select up to 4 relevant facts from the candidate list in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
- If no facts are relevant, return an empty list, e.g., {"fact": []}.
- Only use facts from the candidate list; do NOT generate new facts.
"""

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)


def load_custom_dataset():
    cache = pickle.load(open('data/linker_training/sentence_triple_cache.pkl', 'rb'))

    training = []
    validation = []
    test = []

    gritlm_model = GritLM('GritLM/GritLM-7B', torch_dtype='auto')
    retriever_instruction = 'Given a query, retrieve relevant facts that contribute to answering this query.'
    for file_name in os.listdir('data/linker_training/queries'):
        data = json.load(open(f'data/linker_training/queries/{file_name}', 'r'))
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

    dataset_dict = DatasetDict(datasets)
    return dataset_dict


if __name__ == '__main__':
    model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    dataset = load_custom_dataset()

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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=1024,
        output_dir="exp/fact_linker",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_gpu_eval_batch_size=8
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model("exp/fact_linker/model")
    tokenizer.save_pretrained("exp/fact_linker/tokenizer")
