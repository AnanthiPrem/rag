"""
ingest.py
---------
Loads documents from the ./documents folder (PDF, DOCX, XLSX, TXT, CSV),
splits them into chunks, embeds them locally, and stores them in a
persistent Chroma vector database (./db).

Run this once initially, and again any time you add/update documents.

Usage:
    python ingest.py
"""

import os
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai import types

DOCS_DIR = Path("./documents")
DB_DIR = "./db"
EMBEDDING_MODEL = "gemini-embedding-001"


class GeminiEmbeddings(Embeddings):
    """Embeddings backed by the Gemini API instead of a local model.

    Using an API call instead of a local sentence-transformers model means
    no torch/transformers dependency — those packages are large (1GB+) and
    make cloud deployments slow to build or run out of memory on free-tier
    hosting. This trades a small amount of latency per request for a much
    lighter deployment footprint.
    """

    def __init__(self, api_key, model=EMBEDDING_MODEL, batch_size=100):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend([e.values for e in response.embeddings])
        return vectors

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values


def load_excel(path):
    """Load an Excel file into one Document per sheet using pandas.

    This avoids UnstructuredExcelLoader, which tries to download NLTK data
    from the internet on first use and fails with a 403/network error on
    restricted networks (e.g. corporate firewalls).

    Each sheet's Document starts with a precomputed summary (total row
    count + breakdown of any column that looks categorical, e.g. a
    "Department" column). LLMs are unreliable at accurately counting or
    tallying long lists of rows from raw text, so precomputing these
    numbers here means "how many total / how many per department"
    questions get answered from an exact figure instead of the model
    trying to count text rows itself.
    """
    docs = []
    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue

        summary_lines = [f"Sheet: {sheet_name}", f"Total rows: {len(df)}"]
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            # Treat as categorical if a handful of values repeat often
            # (e.g. a department/status/batch column), not a unique ID column.
            if 1 < nunique <= 30 and nunique < len(df) * 0.5:
                counts = df[col].value_counts(dropna=True)
                breakdown = ", ".join(f"{val}: {cnt}" for val, cnt in counts.items())
                summary_lines.append(f"Breakdown by '{col}': {breakdown}")
        summary_text = "\n".join(summary_lines)

        # Row-level detail as explicit key: value pairs per row — more
        # reliable for an LLM to parse correctly than a whitespace-aligned
        # table dump, especially once it's split into smaller chunks.
        row_lines = [
            " | ".join(f"{col}: {row[col]}" for col in df.columns)
            for _, row in df.iterrows()
        ]
        detail_text = "\n".join(row_lines)

        full_text = f"{summary_text}\n\nRow details:\n{detail_text}"
        docs.append(Document(page_content=full_text, metadata={"sheet": sheet_name}))
    return docs


# Map file extensions to their loader
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}
# Extensions handled by a custom function instead of a LangChain loader class
CUSTOM_LOADERS = {
    ".xlsx": load_excel,
    ".xls": load_excel,
}


def load_single_file(path):
    """Load one file into LangChain Documents. Returns (docs, error_message).

    error_message is None on success.
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in CUSTOM_LOADERS:
            loaded = CUSTOM_LOADERS[ext](str(path))
        elif ext in LOADER_MAP:
            loader = LOADER_MAP[ext](str(path))
            loaded = loader.load()
        else:
            return [], f"Unsupported file type '{ext}'"

        for d in loaded:
            d.metadata["source"] = path.name
        return loaded, None
    except Exception as e:
        return [], str(e)


def load_documents():
    """Walk the documents folder and load every supported file.

    Returns (docs, failures) where failures is a list of
    (filename, error_message) tuples for files that could not be loaded.
    """
    docs = []
    failures = []
    if not DOCS_DIR.exists():
        print(f"Creating {DOCS_DIR} — put your files there and re-run.")
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        return docs, failures

    for path in DOCS_DIR.rglob("*"):
        if path.is_dir():
            continue
        loaded, error = load_single_file(path)
        if error:
            print(f"Failed to load {path.name}: {error}")
            failures.append((path.name, error))
        else:
            docs.extend(loaded)
            print(f"Loaded {path.name} ({len(loaded)} section(s))")
    return docs, failures


def reset_index():
    """Delete all uploaded documents and the vector store, for a clean start."""
    import shutil
    import gc
    import time

    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(DB_DIR):
        # Release any open Chroma client/handles before deleting its files.
        # On Windows, files still memory-mapped by a live client can't be
        # deleted (PermissionError: WinError 32), so clear the cache, force
        # garbage collection, and retry briefly if needed.
        _clear_chroma_cache()
        gc.collect()

        for attempt in range(5):
            try:
                shutil.rmtree(DB_DIR)
                break
            except PermissionError:
                gc.collect()
                time.sleep(0.5)
        else:
            raise PermissionError(
                f"Could not delete {DB_DIR} — it may still be open in another "
                "process. Try restarting the app (stop it with Ctrl+C and "
                "run 'python run_app.py' again) and then use Start fresh."
            )

    _clear_chroma_cache()


def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _clear_chroma_cache():
    """Clear ChromaDB's in-process client cache.

    ChromaDB caches a "system" object per persist_directory path for the
    lifetime of the Python process. In a long-running Streamlit app this
    cache can go stale (especially after reset_index() deletes ./db),
    causing a KeyError on the path when a new client is created. Clearing
    it first avoids that.
    """
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass  # older/newer chromadb versions may not have this method


def build_vector_store(chunks, api_key):
    """Embed chunks via the Gemini API and persist them to Chroma."""
    _clear_chroma_cache()

    embeddings = GeminiEmbeddings(api_key=api_key)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    return vectordb


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set your GEMINI_API_KEY environment variable (or put it in a .env file) before running.")
        print("Get a free key at https://aistudio.google.com/apikey")
        return

    print("Loading documents...")
    docs, failures = load_documents()

    if failures:
        print(f"\n{len(failures)} file(s) failed to load:")
        for name, err in failures:
            print(f"  - {name}: {err}")

    if not docs:
        print("\nNo documents found. Add files to ./documents and re-run.")
        return

    print(f"\nLoaded {len(docs)} document section(s). Splitting into chunks...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("\nEmbedding via Gemini API and storing in Chroma...")
    build_vector_store(chunks, api_key)
    print(f"\nDone. Vector store saved to {DB_DIR}/")
    print("You can now run: python chat.py")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
