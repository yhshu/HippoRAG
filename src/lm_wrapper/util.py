def init_embedding_model(model_name):
    if 'GritLM/' in model_name:
        from src.lm_wrapper.gritlm import GritLMWrapper
        return GritLMWrapper(model_name)
    elif model_name.startswith("Alibaba-NLP/gte-Qwen"):
        from src.lm_wrapper.sentence_transformers_util import SentenceTransformersWrapper
        return SentenceTransformersWrapper(model_name)
    elif model_name.startswith('text-embedding-'):  # OpenAI text embedding models
        from src.lm_wrapper.text_embedding_util import OpenAITextEmbeddingWrapper
        return OpenAITextEmbeddingWrapper(model_name)
    elif model_name not in ['colbertv2', 'bm25']:
        from src.lm_wrapper.huggingface_util import HuggingFaceWrapper
        return HuggingFaceWrapper(model_name)  # HuggingFace model for retrieval


def openai_batch_create_api(file_name, jsonl_contents):
    # Save to the batch file
    assert len(jsonl_contents) > 0
    with open(file_name, 'w') as f:
        f.write('\n'.join(jsonl_contents))
    print("Batch file saved to", file_name, 'len: ', len(jsonl_contents))

    # Call OpenAI Batch API
    from openai import OpenAI
    client = OpenAI(max_retries=5, timeout=60)
    batch_input_file = client.files.create(file=open(file_name, 'rb'), purpose='batch')
    batch_obj = client.batches.create(input_file_id=batch_input_file.id, endpoint='/v1/chat/completions',
                                      completion_window='24h',
                                      metadata={'description': f"Linking dataset synthesis, len: {len(jsonl_contents)}"})
    print(batch_obj)
    print()
    print("Go to https://platform.openai.com/batches/ or use OpenAI file API to get the output file ID after the batch job is done.")
