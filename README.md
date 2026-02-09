# 🧠 AI Academic Assistant
![Banner](assets/banner.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

**Your personal, local research companion.**  
Discover, manage, and analyze academic literature with the power of local LLMs.
No cloud dependency, no privacy compromise.

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

| Core Capabilities | Advanced Analysis | Management |
| :--- | :--- | :--- |
| **🔍 Smart Search**<br>Semantic search across arXiv and your local library. | **💡 Idea Generator**<br>Brainstorm new research directions based on your library. | **📂 Library Manager**<br>Tag, filter, and organize papers. |
| **📥 One-Click Ingest**<br>Download and index papers from arXiv instantly. | **✍️ Style Polish**<br>Rewrite sentences in professional academic styles. | **🏷️ Metadata Editor**<br>Fix titles, authors, and DOIs easily. |
| **📝 Summarization**<br>Generate concise cards for quick reading. | **📊 Graph Analytics**<br>Visualize citation networks and co-authorships. | **📤 Rich Export**<br>BibTeX, RIS, Zotero JSON support. |

---

## 🛠️ Technology Stack

Built with modern, powerful open-source tools:

*   **UI**: [Streamlit](https://streamlit.io/)
*   **LLM Orchestration**: [LangChain](https://www.langchain.com/)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/)
*   **Local Inference**: [Ollama](https://ollama.com/)

---

## 📦 Detailed Installation

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com) (for local LLM inference)

### Setup Steps
1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/arxiv-assistant.git
    cd arxiv_assistant
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
    Required for sentence analysis.
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
