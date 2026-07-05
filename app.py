"""
app.py
------
Streamlit UI for the internal document RAG chatbot.

Usage:
    streamlit run app.py

Then open the URL it prints (usually http://localhost:8501) in your browser.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

import ingest
import chat

load_dotenv()


def get_default_api_key():
    """Check local .env first, then Streamlit Cloud's secrets store."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass  # no secrets.toml configured — fine for local use
    return key


st.set_page_config(page_title="Internal Docs Chatbot", page_icon="📄", layout="wide")

st.title("📄 Internal Document Chatbot")
st.caption("Ask questions about your internal documents (RAG-powered)")

# ---------------- Sidebar: setup ----------------
with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Gemini API Key",
        value=get_default_api_key(),
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )

    st.divider()
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls", "txt", "csv"],
        accept_multiple_files=True,
    )

    upload_mode = st.radio(
        "When processing",
        ["Replace existing documents", "Add to existing documents"],
        index=0,
        help="Replace clears out anything indexed before and starts fresh with just the files you upload now.",
    )

    if st.button("Process documents", use_container_width=True):
        if not api_key:
            st.warning("Enter your Gemini API key above first — it's needed to create embeddings now, not just to chat.")
        elif not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            if upload_mode == "Replace existing documents":
                st.session_state.pop("vectordb", None)  # release open db handles first
                ingest.reset_index()

            ingest.DOCS_DIR.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            for f in uploaded_files:
                dest = ingest.DOCS_DIR / f.name
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(dest)

            with st.spinner("Loading, chunking, and embedding document(s) via Gemini..."):
                docs = []
                failures = []
                for path in saved_paths:
                    loaded, error = ingest.load_single_file(path)
                    if error:
                        failures.append((path.name, error))
                    else:
                        docs.extend(loaded)

                chunks = ingest.chunk_documents(docs) if docs else []
                embed_error = None
                if docs:
                    try:
                        ingest.build_vector_store(chunks, api_key)
                    except Exception as e:
                        embed_error = str(e)
                        docs = []  # don't report success below

            if embed_error:
                if "429" in embed_error or "RESOURCE_EXHAUSTED" in embed_error:
                    st.error(
                        "Gemini's free tier rate limit was hit while embedding your "
                        "documents (this app already retries automatically, but the "
                        "limit was hit repeatedly). Wait a minute and try again with "
                        "fewer/smaller files at a time, or check your quota at "
                        "https://ai.dev/rate-limit."
                    )
                else:
                    st.error(f"Failed to embed documents: {embed_error}")

            if failures:
                st.error(f"{len(failures)} file(s) failed to process:")
                for name, err in failures:
                    st.write(f"- **{name}**: {err}")

            if docs:
                action = "Indexed" if upload_mode == "Replace existing documents" else "Added"
                st.success(f"{action} {len(saved_paths) - len(failures)} file(s) — {len(chunks)} chunks.")
                st.session_state.pop("vectordb", None)  # force reload on next question
                st.session_state.messages = []  # old chat history referenced the old document set
            elif not failures:
                st.warning("No readable content found in the uploaded file(s).")

    # Show what's currently indexed
    if ingest.DOCS_DIR.exists():
        indexed_files = sorted(p.name for p in ingest.DOCS_DIR.iterdir() if p.is_file())
        if indexed_files:
            with st.expander(f"Currently indexed ({len(indexed_files)})"):
                for name in indexed_files:
                    st.write(f"- {name}")

    if st.button("🗑️ Start fresh (clear all documents)", use_container_width=True):
        st.session_state.pop("vectordb", None)  # release open db handles first
        ingest.reset_index()
        st.session_state.pop("vectordb", None)
        st.session_state.messages = []
        st.success("Cleared. Upload new documents to begin again.")
        st.rerun()

    st.divider()
    top_k = st.slider("Chunks retrieved per question", min_value=1, max_value=10, value=chat.TOP_K)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Vector store (cached per session) ----------------
def get_vectordb():
    if not api_key:
        return None
    if "vectordb" not in st.session_state:
        if not os.path.exists(chat.DB_DIR):
            return None
        st.session_state.vectordb = chat.load_vector_store(api_key)
    return st.session_state.vectordb


vectordb = get_vectordb()

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat input ----------------
if vectordb is None:
    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to get started.")
    else:
        st.info("Upload and process documents in the sidebar to get started.")
else:
    query = st.chat_input("Ask a question about your documents...")
    if query:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context, sources = chat.retrieve_context(vectordb, query, k=top_k)
                    if not context:
                        answer = "I couldn't find anything relevant in the documents."
                    else:
                        client = genai.Client(api_key=api_key)
                        answer = chat.ask_gemini(client, query, context)

                    st.markdown(answer)

                    if context:
                        with st.expander("Sources used"):
                            seen = set()
                            for s in sources:
                                src = s.metadata.get("source", "unknown")
                                if src not in seen:
                                    st.write(f"- {src}")
                                    seen.add(src)

            st.session_state.messages.append({"role": "assistant", "content": answer})
