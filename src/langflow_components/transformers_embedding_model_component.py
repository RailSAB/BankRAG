from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, MessageTextInput
import requests
from langchain.embeddings.base import Embeddings
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.io import SecretStrInput, BoolInput
from langflow.field_typing import Embeddings as LCEmbeddingType


class TransformersInferenceEmbeddings(Embeddings):
    def __init__(self, endpoint="http://localhost:8081"):
        self.endpoint = endpoint.rstrip("/")

    def embed_query(self, text: str):
        res = requests.post(f"{self.endpoint}/v1/encode", json={"text": text})
        res.raise_for_status()
        return res.json()["vector"]

    def embed_documents(self, texts: list[str]):
        res = requests.post(f"{self.endpoint}/v1/encode_batch", json={"texts": texts})
        res.raise_for_status()
        return res.json()["vectors"]


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


