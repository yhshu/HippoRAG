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
