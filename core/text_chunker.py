from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    if not documents:
        raise ValueError("No documents were provided for chunking.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the documents.")

    return chunks
