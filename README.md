# Internal Document RAG Chatbot

A local prototype that answers questions using your internal documents
(PDF, DOCX, XLSX, TXT, CSV) via Retrieval-Augmented Generation.

## How it works

1. `ingest.py` loads every file in `./documents`, splits it into chunks,
   embeds those chunks **locally** (no data leaves your machine at this
   step), and stores them in a Chroma vector database in `./db`.
2. `chat.py` takes your question, finds the most relevant chunks from the
   vector store, and sends them to Google Gemini (free tier) along with
   your question so it can answer using only your documents.

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your documents**
   Drop your PDF/DOCX/XLSX/TXT/CSV files into the `documents/` folder.

3. **Set your Gemini API key** (free, no credit card required)
   Copy `.env.example` to `.env` and fill in your key:
   ```bash
   cp .env.example .env
   ```
   Get a free key at https://aistudio.google.com/apikey — sign in with
   any Google account and click "Create API key."

4. **Run the app**

   **Option A — Web UI (recommended):**
   ```bash
   streamlit run app.py
   ```
   This opens a browser tab (usually at http://localhost:8501) where you
   can upload documents, click "Process documents" to build the index,
   and chat in a normal chat interface — no need to run `ingest.py`
   separately first.

   **Option B — Command line:**
   ```bash
   python ingest.py   # build the vector store from files in ./documents
   python chat.py      # then chat in the terminal
   ```

## Notes & things you'll likely want to tune

- **Chunk size/overlap** (`ingest.py`): 1000/150 chars is a reasonable
  default. Smaller chunks = more precise retrieval but less context per
  chunk; larger chunks = more context but noisier retrieval.
- **TOP_K** (`chat.py`): how many chunks get retrieved per question.
  Increase if answers seem to be missing information; decrease if the
  model seems distracted by irrelevant context.
- **Embedding model**: `all-MiniLM-L6-v2` is fast and runs on CPU. For
  better accuracy (at the cost of speed), try `all-mpnet-base-v2`.
- **Scanned/image-only PDFs** won't extract text with `PyPDFLoader` — 
  you'd need OCR (e.g. `pytesseract`) added to the pipeline for those.
- **Very large document sets**: Chroma's fine for prototyping and small-to-
  medium collections. If you outgrow it, look at Qdrant, Weaviate, or
  Pinecone for production-scale deployments.
- **Security**: this is a local prototype. Once you move beyond your own
  machine (a team-wide internal tool), think about access control, since
  the chatbot will surface whatever is in `documents/` to whoever can
  query it.
- **Gemini free tier limits**: the free tier has request-per-minute and
  request-per-day caps that vary by model and change over time. Fine for
  a personal prototype; if you hit a `429` error, you're being rate
  limited — wait a bit and retry, or check current limits at
  https://ai.google.dev/gemini-api/docs/rate-limits.
- **Data privacy note**: on the free tier, Google may use your prompts to
  improve their products (unlike the paid tier, which doesn't). If your
  internal documents are sensitive, keep this in mind, read Google's
  terms at https://ai.google.dev/gemini-api/terms, or consider the paid
  tier / a different provider for real internal data.

## Deploying (e.g. to Streamlit Community Cloud)

**If GitHub rejects your upload for being too large**, it's almost always
the `venv/` folder — it can be several GB because it contains every
installed package (including PyTorch). It should never be committed;
`requirements.txt` is what lets the server reinstall everything fresh.
The `.gitignore` included in this project already excludes `venv/`, `db/`,
`documents/`, and `.env`. If you already committed `venv/` before adding
`.gitignore`, remove it from git's tracking (this won't delete it from
your computer):
```bash
git rm -r --cached venv
git commit -m "Remove venv from version control"
```

**Steps to deploy on Streamlit Community Cloud (free):**
1. Push this project to a GitHub repo (with `.gitignore` in place, so only
   your code — not `venv/`, `db/`, or your documents — gets uploaded)
2. Go to https://share.streamlit.io, sign in, and click "New app"
3. Select your repo and set the main file to `app.py`
4. Under **Advanced settings → Secrets**, add your API key in this format:
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   ```
   (`app.py` checks this automatically — no code changes needed)
5. Deploy. The first load will take a few minutes while it installs
   dependencies and downloads the embedding model.

**Note on documents when deployed:** since `documents/` and `db/` aren't
committed to GitHub, a freshly deployed app starts empty — users upload
and process documents through the UI itself, same as running locally.
Anything processed lives only on that server session; if the app restarts
or redeploys, uploaded documents are cleared and need to be re-uploaded.

## Project structure

```
rag_chatbot/
├── documents/       # put your source files here
├── db/              # generated vector store (created by ingest.py)
├── app.py           # Streamlit web UI (recommended entry point)
├── ingest.py        # loads, chunks, embeds documents
├── chat.py          # command-line Q&A loop
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```
