import os

from dotenv import load_dotenv
from google import genai

from core.retriever import retrieve_documents

load_dotenv()

MODEL_NAME = "gemma-4-26b-a4b-it"


def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    return genai.Client(api_key=api_key)


def build_prompt(question, retrieved_documents):
    context_parts = []

    for document, distance in retrieved_documents:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present in the context, clearly say:
   "I couldn't find the answer in the provided document."
4. Keep the answer concise and factual.
5. When possible, mention the relevant page number.
6. Treat the document context as the only source of truth.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""


def answer_question(vector_store, question, k=4):
    retrieved_documents = retrieve_documents(
        vector_store,
        question,
        k=k,
    )

    prompt = build_prompt(
        question,
        retrieved_documents,
    )

    client = get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    answer = response.text.strip()

    if not answer:
        raise ValueError("The model returned an empty response.")

    sources = []

    for document, distance in retrieved_documents:
        sources.append(
            {
                "source": document.metadata.get(
                    "source",
                    "Unknown source",
                ),
                "page": document.metadata.get(
                    "page",
                    "Unknown page",
                ),
                "distance": float(distance),
                "content": document.page_content,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }
