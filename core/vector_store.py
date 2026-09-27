from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from core.embeddings import get_embeddings


class FastEmbedAdapter(Embeddings):
    def __init__(self):
        self.model = get_embeddings()

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


def create_vector_store(documents):
    if not documents:
        raise ValueError("No document chunks were provided.")

    embeddings = FastEmbedAdapter()

    return FAISS.from_documents(
        documents,
        embeddings,
    )
