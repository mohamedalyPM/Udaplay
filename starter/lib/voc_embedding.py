# starter/lib/voc_embedding.py
import os
from typing import List
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI

class VocareumEmbeddingFunction(EmbeddingFunction):
   
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VOC_OPENAI_API_KEY")
        assert api_key and api_key.startswith("voc-"), "Missing voc- OPENAI_API_KEY"
        self.client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
        self.model_name = model_name

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]
