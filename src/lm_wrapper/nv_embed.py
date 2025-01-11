# https://huggingface.co/nvidia/NV-Embed-v2
import sys
sys.path.append('.')

from typing import Union, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.lm_wrapper import EmbeddingModelWrapper

class NVEmbedV2Wrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = 'nvidia/NV-Embed-v2', max_seq_length: int = 2048, multi_gpu=False):
        # Initialize the model with specified configurations
        device = 'cuda' if multi_gpu is False else 'cpu'
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = max_seq_length
        self.model.tokenizer.padding_side = "right"
        self.multi_gpu = multi_gpu
        print(f'Initialized NV-Embed-v2 model, multi-GPU: {multi_gpu}, max_seq_length: {max_seq_length}, device: {device}')

    def _add_eos(self, input_examples: List[str]) -> List[str]:
        # Adds EOS token to each example
        return [example + self.model.tokenizer.eos_token for example in input_examples]

    def encode_list(self, texts: List[str], instruction: str, batch_size: int = 2) -> torch.Tensor:
        # Encode the list of texts with instruction as prefix
        if instruction is not None and instruction != '':
            prompt = f"Instruct: {instruction}\nQuery: "
        else:
            prompt = None
        print(f'NV-Embed-v2 encoding, batch size: {batch_size}, text len: {len(texts)}')
        try:
            embeddings = self.model.encode(self._add_eos(texts), batch_size=batch_size, prompt=prompt, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            print('Error in encode_list:', e)
            print('Type of texts:', type(texts))
            print('Len of texts:', len(texts))
            print('Batch size:', batch_size)

    def encode_list_multi_gpu(self, texts: List[str], instruction: str, batch_size: int) -> torch.Tensor:
        # Encode the list of texts with instruction as prefix
        if instruction is not None and instruction != '':
            prompt = f"Instruct: {instruction}\nQuery: "
        else:
            prompt = None
        print(f'NV-Embed-v2 encoding, batch size per GPU: {batch_size}, #GPU: {torch.cuda.device_count()}, len to encode: {len(texts)}')
        try:
            pool = self.model.start_multi_process_pool()
            emb = self.model.encode_multi_process(self._add_eos(texts), pool, prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
            self.model.stop_multi_process_pool(pool)
        except Exception as e:
            print('Error in encode_list_multi_gpu:', e)
            print('Type of texts:', type(texts))
            print('Len of texts:', len(texts))
            print('Batch size:', batch_size)
        return emb

    def encode_text(self, text: Union[str, List[str]], instruction: str = '', norm: bool = True, return_cpu: bool = False, return_numpy: bool = False, batch_size=16) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        if self.multi_gpu:
            embeddings = self.encode_list_multi_gpu(text, instruction, batch_size=batch_size)
        else:
            embeddings = self.encode_list(text, instruction, batch_size=batch_size)

        if norm:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T
            if isinstance(embeddings, np.ndarray):
                embeddings = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T
        if isinstance(embeddings, torch.Tensor):
            if return_cpu:
                embeddings = embeddings.cpu()
            if return_numpy:
                embeddings = embeddings.numpy()

        return embeddings

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        # Calculate similarity scores between query and document vectors
        return np.dot(query_vec, doc_vecs.T)

if __name__ == '__main__':
    queries = [
        'are judo throws allowed in wrestling?',
        'how to become a radiology technician in michigan?'
    ]

    # No instruction needed for retrieval passages
    passages = [
        "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
        "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
    ]

    embedding_model = NVEmbedV2Wrapper()
    query_embeddings = embedding_model.encode_text(queries, instruction="Given a question, retrieve passages that answer the question", return_numpy=True)
    passage_embeddings = embedding_model.encode_text(passages, instruction="", return_numpy=True)
    scores = embedding_model.get_query_doc_scores(query_embeddings, passage_embeddings)
    print(scores)
