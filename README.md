# Local Academic Assistant - Usage Guide

## Setup

1.  **Install Dependencies**:
    Open a terminal in `p:\workspace\arxiv_assistant` and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install & Run Ollama**:
    - Download from [ollama.com](https://ollama.com).
    - Run `ollama run llama3` (or `mistral`) in a separate terminal to download and serve the model.
    - Ensure it is listening on port 11434 (default).

## Running the App

1.  Run the Streamlit app:
    ```bash
    streamlit run src/app.py
    ```
2.  The browser will open automatically.

## Features

1.  **Manage Database**: Use the sidebar to search arXiv for topics (e.g., "Mechanism Design") and download papers.
2.  **Search Literature**: In the "Semantic Search" tab, ask questions or search for concepts.
3.  **Academic Sentence Polisher**: In the "Rewrite" tab, paste a drafted sentence. The system will retrieve similar sentences from your downloaded papers and use the LLM to rewrite yours in a better style.
4.  **Research Idea Generator**: Enter a topic, and the system will read your library to suggest new directions.
