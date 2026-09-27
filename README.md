# DocMind

**Ask questions. Query documents. Get grounded answers.**

DocMind is a PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. Upload a document, ask questions about it, and get answers grounded in the document with the relevant source pages and retrieved content.

## Live Demo

[DocMind](https://docmind-pdf-explainer.streamlit.app)

## Features

- PDF upload and text extraction
- Document chunking with page metadata
- Local embeddings using **FastEmbed**
- Vector search using **FAISS**
- Grounded answers using **Google Gemma**
- Retrieved source pages and passages
- Multi-question conversation history
- Document statistics
- Conversation download
- Clear and re-index documents
- Handles questions whose answers are not found in the document

## RAG Pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Local Embeddings
 ↓
FAISS Vector Search
 ↓
Relevant Context
 ↓
Google Gemma
 ↓
Grounded Answer + Sources
```
Tech Stack
Python 3.12 · Streamlit · Google GenAI · Gemma 4 · FastEmbed · FAISS · LangChain · PyPDF · uv


Author
Hasshya Krishnamoorthy



