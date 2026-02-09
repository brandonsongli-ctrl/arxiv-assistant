# 🧠 AI Academic Assistant (学术助手)
![Banner](assets/banner.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

**您的本地个性化研究伙伴。**  
利用本地 LLM 的强大功能，发现、管理和分析学术文献。
无云端依赖，无隐私风险。

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 🚀 快速开始

几分钟内即可启动并运行。

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **安装并启动 Ollama**
    访问 [ollama.com](https://ollama.com) 下载并安装 Ollama。
    安装完成后，在终端运行以下命令以拉取并启动模型：
    ```bash
    ollama run llama3
    ```

3.  **启动应用**
    ```bash
    streamlit run src/app.py
    ```

---

## ✨ 功能特性

| 核心能力 | 高级分析 | 管理工具 |
| :--- | :--- | :--- |
| **🔍 智能检索**<br>在 arXiv 和本地文献库中进行语义检索。 | **💡 灵感生成**<br>基于您的文献库构思新的研究方向。 | **📂 文献库管理**<br>标记、筛选和组织论文。 |
| **📥 一键导入**<br>即时下载 arXiv 论文并建立索引。 | **✍️ 风格润色**<br>将句子改写为专业的学术风格。 | **🏷️ 元数据编辑**<br>轻松修复标题、作者和 DOI。 |
| **📝 摘要生成**<br>生成简洁的论文摘要卡片，快速阅读。 | **📊 图谱分析**<br>可视化引用网络和共同作者关系。 | **📤 丰富导出**<br>支持 BibTeX, RIS, Zotero JSON 导出。 |

---

## 🛠️ 技术栈

基于现代、强大的开源工具构建：

*   **UI**: [Streamlit](https://streamlit.io/)
*   **LLM 编排**: [LangChain](https://www.langchain.com/)
*   **向量数据库**: [ChromaDB](https://www.trychroma.com/)
*   **本地推理**: [Ollama](https://ollama.com/)

---

## 📦 详细安装说明

### 前置要求
*   Python 3.10+
*   [Ollama](https://ollama.com) (用于本地 LLM 推理)

### 安装步骤
1.  **克隆仓库**
    ```bash
    git clone https://github.com/yourusername/arxiv-assistant.git
    cd arxiv_assistant
    ```

2.  **安装 Ollama**
    用于运行本地大语言模型。
    *   访问 [ollama.com](https://ollama.com) 下载并安装。
    *   验证安装：`ollama --version`
    *   拉取模型：`ollama run llama3`

3.  **安装 Python 依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **安装 Spacy 模型**
    用于句子分析。
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **可选：OCR 支持**
    用于扫描版 PDF（无文本层）。
    *   并在系统安装 `poppler` 和 `tesseract`。
    *   安装 Python 依赖：`pip install pdf2image pytesseract`
    *   在 `.env` 中设置 `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK=1`。

---

## ⚙️ 配置

通过环境变量或 `.env` 文件自定义您的体验。

| 变量名 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `ARXIV_ASSISTANT_DB_DIR` | ChromaDB 存储路径 | `./data/chroma_db` |
| `ARXIV_ASSISTANT_EMBEDDING_MODEL` | Embedding 模型 | `all-MiniLM-L6-v2` |
| `ARXIV_ASSISTANT_RETRIEVAL_MODE` | 检索模式 (vector/bm25/hybrid) | `hybrid` |
| `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK` | 启用扫描版 PDF OCR | `0` |

*查看 `src/config.py` 获取所有可用选项。*

---

## 🤝 贡献

欢迎提交 Pull Request 来改进本项目！
