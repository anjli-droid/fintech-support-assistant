# FinTech Customer Support Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers customer questions
about financial products (loans, fixed deposits, mutual funds) using only the
company's policy PDFs. Every answer ends with a source line showing the
document title and section.

**Live app:** [your Streamlit URL]

## Features
- Answers grounded only in the provided PDFs (no guessing)
- Source line with document title and section for every answer
- Chat history, so follow-up questions work
- Fixed "I could not find this information" reply for questions outside the documents

## RAG Architecture

```
PDFs (Dataset/) -> load pages -> clean text -> add document title
   -> chunk (900 chars, 100 overlap) -> embed (MiniLM) -> FAISS index

Question -> embed -> top 4 chunks (FAISS) -> prompt (context + last 6 messages)
   -> LLM (Groq) -> answer + source
```

1. **Preprocessing:** the rupee sign (extracted as a square block) is fixed, markdown bold markers are removed, and each PDF's title is read from its first page and attached to every page as metadata.
2. **Chunking:** RecursiveCharacterTextSplitter, 900 characters with 100 overlap.
3. **Embedding and storage:** each chunk is converted to a vector and stored in a FAISS index, built once per session.
4. **Retrieval:** the question is embedded and the 4 most similar chunks are returned.
5. **Generation:** the chunks and the last 6 chat messages go into a strict prompt. The LLM must answer only from the context, reply with a fixed message if the answer is missing, and end with a `Source:` line.

## Models and Vector Database
| Component | Choice |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dimensions) |
| Vector database | FAISS (in-memory, built at startup) |
| LLM | `openai/gpt-oss-120b` on Groq, temperature 0 |
| Chunking | RecursiveCharacterTextSplitter, size 900, overlap 100 |
| Retrieval | Similarity search, k = 4 |
| Framework / UI | LangChain / Streamlit |

## Setup Instructions
1. Clone the repository:
```bash
   git clone [your repo URL]
   cd [repo folder name]
```
2. Create and activate a virtual environment (Python 3.11 or 3.12 recommended):
```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Create a `.env` file (see Environment Variables below).
5. Put your PDF files in the `Dataset/` folder.
6. Run the app:
```bash
   streamlit run app.py
```

## Environment Variables
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com/keys |

Copy `.env.example` to `.env` and add your own key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## API Documentation
Not applicable. The app is a Streamlit web interface and exposes no API
endpoint of its own. It calls the Groq API internally for the LLM.


