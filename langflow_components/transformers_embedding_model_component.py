from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, MessageTextInput
import requests
from langchain.embeddings.base import Embeddings
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.io import SecretStrInput, BoolInput
from langflow.field_typing import Embeddings as LCEmbeddingType


class TransformersInferenceEmbeddings(Embeddings):
    def __init__(self, endpoint="http://localhost:8081", timeout=10):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

    def _post(self, text):
        text = text.strip()
        if not text:
            return [0.0] * 384
        payload = {"text": text, "config": {}}
        r = requests.post(f"{self.endpoint}/vectors", json=payload,
                          headers=self.headers, timeout=self.timeout)
        if r.status_code != 200:
            raise Exception(f"Embedding server error {r.status_code}: {r.text}")
        return r.json()["vector"]

    def embed_query(self, text: str):
        return self._post(text)

    def embed_documents(self, texts):
        return [self._post(t) for t in texts]


class WeaviateTransformersEmbeddingModelComponent(LCEmbeddingsModel):
    display_name = "Weaviate Transformers Embedding"
    description = "Embeddings using Weaviate's text2vec-transformers inference container."
    name = "WeaviateTransformersEmbedding"
    icon = "binary"
    category = "models"

    inputs = [
        SecretStrInput(
            name="endpoint",
            display_name="Transformers API Endpoint",
            info="URL of the text2vec-transformers container.",
        ),
        BoolInput(
            name="batch_mode",
            display_name="Use Batch Encoding",
            value=True,
            advanced=True,
            info="Use /v1/encode_batch endpoint for multiple documents.",
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        endpoint = self.endpoint
        return TransformersInferenceEmbeddings(endpoint=endpoint)


