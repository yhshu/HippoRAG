import numpy as np
from sentence_transformers import SentenceTransformer

from src.lm_wrapper import EmbeddingModelWrapper


class SentenceTransformersWrapper(EmbeddingModelWrapper):

    def __init__(self, model_name, max_seq_length=1024):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if model_name.startswith("Alibaba-NLP/gte-Qwen2"):
            self.model.max_seq_length = max_seq_length

    def encode_text(self, text, instruction=None, norm=True, return_cpu=True, return_numpy=True, batch_size=32):
        if isinstance(text, str):
            text = [text]
        if self.model_name.startswith("Alibaba-NLP/gte-Qwen2"):
            return self.model.encode(text, prompt_name=instruction, batch_size=batch_size)


if __name__ == '__main__':
    model = SentenceTransformersWrapper("Alibaba-NLP/gte-Qwen2-7B-instruct")

    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    query_embeddings = model.encode_text(queries, 'query')
    document_embeddings = model.encode_text(documents)

    scores = np.dot(document_embeddings, query_embeddings.T)
    print(scores.tolist())
