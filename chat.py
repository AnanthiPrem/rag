"""
chat.py
-------
Interactive RAG chatbot. Retrieves relevant chunks from the Chroma vector
store built by ingest.py, then asks Gemini to answer using only that
context.

Usage:
    python chat.py
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from google import genai

from ingest import GeminiEmbeddings, DB_DIR as INGEST_DB_DIR

load_dotenv()

DB_DIR = INGEST_DB_DIR
TOP_K = 4  # number of chunks to retrieve per question
MODEL = "gemini-2.5-flash"  # free-tier eligible; swap for another Gemini model if needed

SYSTEM_PROMPT = """You are an internal knowledge assistant. Answer the user's \
question using ONLY the provided document excerpts below. If the excerpts \
don't contain enough information to answer, say so clearly instead of \
guessing. Always mention which source document(s) you drew from.
"""


def _clear_chroma_cache():
    """Clear ChromaDB's in-process client cache (see ingest.py for why)."""
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def load_vector_store(api_key):
    _clear_chroma_cache()
    embeddings = GeminiEmbeddings(api_key=api_key)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def retrieve_context(vectordb, query, k=TOP_K):
    results = vectordb.similarity_search(query, k=k)
    context_blocks = []
    for r in results:
        source = r.metadata.get("source", "unknown")
        context_blocks.append(f"[Source: {source}]\n{r.page_content}")
    return "\n\n---\n\n".join(context_blocks), results


def ask_gemini(client, question, context):
    user_message = f"""{SYSTEM_PROMPT}

Document excerpts:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
    )
    return response.text


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set your GEMINI_API_KEY environment variable (or put it in a .env file) before running.")
        print("Get a free key at https://aistudio.google.com/apikey")
        return

    if not os.path.exists(DB_DIR):
        print("No vector store found. Run 'python ingest.py' first.")
        return

    print("Loading vector store...")
    vectordb = load_vector_store(api_key)
    client = genai.Client(api_key=api_key)

    print("\nRAG chatbot ready. Ask a question about your documents (type 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        context, sources = retrieve_context(vectordb, query)
        if not context:
            print("Assistant: I couldn't find anything relevant in the documents.\n")
            continue

        answer = ask_gemini(client, query, context)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
