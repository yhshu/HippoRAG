import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.lm_wrapper import EmbeddingModelWrapper


class OpenAITextEmbeddingWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_name_processed = model_name.replace('/', '_').replace('.', '_')
        self.client = OpenAI(max_retries=5, timeout=60)

    def encode_text(self, text, instruction='', norm=False, return_cpu=False, return_numpy=True) -> np.ndarray:
        if instruction is None:
            instruction = ''
        if isinstance(text, list):
            texts = [instruction + t for t in text]
        elif isinstance(text, str):
            texts = [instruction + text]
        else:
            texts = []

        embeddings = []
        if len(texts) <= 1:
            response = self.client.embeddings.create(
                input=texts[0],
                model=self.model_name,
            )
            embeddings.append(response.data[0].embedding)
        else:
            for t in tqdm(texts, total=len(texts), desc='OpenAI embedding encoding text'):
                response = self.client.embeddings.create(
                    input=t,
                    model=self.model_name,
                )
                embeddings.append(response.data[0].embedding)

        if return_numpy:
            embeddings = np.array(embeddings)
        return embeddings


if __name__ == '__main__':
    model = OpenAITextEmbeddingWrapper("text-embedding-3-small")
    query = "who is the president of the united states?"
    query_embedding = model.encode_text(query)

    candidates = ['Donald Trump', 'Joe Biden', 'Kamala Harris']
    doc_embeddings = model.encode_text(candidates)

    scores = np.dot(doc_embeddings, query_embedding.T)
    print(scores)
