def retrieve_documents(vector_store, question, k=4):
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    results = vector_store.similarity_search_with_score(
        question.strip(),
        k=k,
    )

    if not results:
        raise ValueError("No relevant documents were found.")

    return results
