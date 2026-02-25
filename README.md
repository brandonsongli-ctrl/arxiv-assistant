# 🧠 AI Academic Assistant
![Banner](assets/banner.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

**Your personal, local-first research companion.**  
A powerful toolkit for discovering, managing, and analyzing academic literature using local LLMs, vector databases, and rich academic APIs (Semantic Scholar, OpenAlex, arXiv). Build your personal knowledge base with zero cloud dependency and zero privacy compromise.

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 🚀 Quick Start

Get up and running in minutes.

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **Install & Start Ollama**
    Download and install [Ollama](https://ollama.com).
    Then run the following command to pull and start the model:
    ```bash
    ollama run llama3
    ```

3.  **Launch App**
    ```bash
    streamlit run src/app.py
    ```

---

## ✨ Features

### 🔍 Advanced Literature Search & RAG
- **Hybrid Semantic Search**: Uses ChromaDB and BM25 to instantly search across your local library and external academic databases.
- **Academic API Integrations**: Seamlessly fetches papers and metadata from **arXiv**, **Semantic Scholar**, and **OpenAlex**.
- **Chat with Papers**: Use Retrieval-Augmented Generation (RAG) powered by local LLMs to ask questions directly against your ingested documents.

### 📂 Intelligent Library Management
- **Automated Metadata Enrichment**: Automatically fills in missing DOIs, authors, and publication years using external academic graphs.
- **Smart Deduplication**: Identifies and merges duplicate papers in your library to keep your database clean.
- **Tagging & Filtering**: Organize papers intuitively with tags and robust filtering mechanisms.
- **Rich Export**: Export your library to BibTeX, RIS, or Zotero-compatible JSON.

### 📊 Powerful Analytics & Insights
- **Citation Graph Analytics**: Visualize citation networks, co-authorships, and reference links directly inside the app.
- **Idea Generator**: Let the local LLM brainstorm new research directions based on your curated reading list.
- **Style Polish**: Rewrite and polish your academic sentences into professional literature styles.
- **Automated Summarization**: Generate concise summary cards for rapid literature review.

---

## 🛠️ Technology Stack

Built with modern, powerful open-source tools:

*   **UI Framework**: [Streamlit](https://streamlit.io/)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) for fast semantic retrieval
*   **LLM Orchestration**: [LangChain](https://www.langchain.com/) for RAG and text processing
*   **Local Inference**: [Ollama](https://ollama.com/) (privacy-first LLM usage)
*   **Data Enrichment**: APIs from Semantic Scholar, OpenAlex, and arXiv.

---

## 📦 Detailed Installation

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com) (for local LLM inference)

### Setup Steps
1.  **Clone the repository**
    ```bash
    git clone https://github.com/brandonsongli-ctrl/arxiv-assistant.git
    cd arxiv-assistant
    ```

2.  **Install & Start Ollama**
    Required for local LLM inference.
    *   Download and install from [ollama.com](https://ollama.com).
    *   Verify installation: `ollama --version`
    *   Pull model: `ollama run llama3`

3.  **Install Python requirements**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Spacy Model**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Optional: OCR Support**
    For scanning older PDFs without text layers.
    *   Install `poppler` and `tesseract` on your system.
    *   Install python deps: `pip install pdf2image pytesseract`
    *   Set `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK=1` in `.env`.

---

## ⚙️ Configuration

Customize your experience via environment variables or a `.env` file.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ARXIV_ASSISTANT_DB_DIR` | ChromaDB storage path | `./data/chroma_db` |
| `ARXIV_ASSISTANT_EMBEDDING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `ARXIV_ASSISTANT_RETRIEVAL_MODE` | Search mode (vector/bm25/hybrid) | `hybrid` |
| `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK` | Enable OCR for scanned PDFs | `0` |

*See `src/config.py` for all available options.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
