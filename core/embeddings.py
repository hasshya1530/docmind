from fastembed import TextEmbedding


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
    )


def embed_documents(documents):
    if not documents:
        raise ValueError("No documents were provided for embedding.")

    embeddings = get_embeddings()
    texts = [document.page_content for document in documents]

    return list(embeddings.embed(texts))


def embed_query(query):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    embeddings = get_embeddings()

    return list(
        embeddings.embed(
            [query.strip()]
        )
    )[0]
