# 🧠 AI 学术助手 (AI Academic Assistant)
![Banner](assets/banner.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

**您首选的、本地化优先的智能科研伴侣。**  
通过强大的本地大语言模型 (LLM)、向量数据库，以及丰富的学术 API 接口 (Semantic Scholar, OpenAlex, arXiv)，为您提供一个全方位的学术文献发现、管理和深层分析工具。构建专属您的个人知识图谱：**无需依赖云服务，彻底保护数据隐私**。

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 🚀 快速开始

只需几分钟，即可搭建和运行您的本地学术中心。

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **安装并启动 Ollama**
    访问 [ollama.com](https://ollama.com) 下载并安装 Ollama。
    安装完成后，运行以下命令拉取并启动模型：
    ```bash
    ollama run llama3
    ```

3.  **启动应用**
    ```bash
    streamlit run src/app.py
    ```

---

## ✨ 核心特性

### 🔍 高级文献检索与 RAG 问答
- **混合语义检索**: 融合 ChromaDB 向量检索与 BM25，在您的本地文献库及外部学术网络中实现毫秒级的精准搜索。
- **学术 API 深度集成**: 流畅对接并抓取来自 **arXiv**、**Semantic Scholar** 和 **OpenAlex** 的文献与元数据。
- **基于文档的对话**: 利用本地大语言模型 (LLMs) 构建强大的 RAG（检索增强生成）系统，直接向您的论文库提问，获取精准溯源的回答。

### 📂 智能文献库管理
- **自动化元数据补全**: 自动接入外部知识图谱补齐论文缺失的 DOI、作者和出版年份等核心元数据。
- **智能去重**: 智能识别并合并文献库中的重复论文，保持数据库干净整洁。
- **标签与过滤系统**: 使用灵活的标签和条件过滤机制，直观管理您的科研文献。
- **丰富的导出选项**: 轻松将文献导出为 BibTeX, RIS, 或者 Zotero 兼容的 JSON 格式。

### 📊 强大的数据分析与学术洞见
- **引用网络图谱分析**: 在应用内直观可是化并探索引用网络、作者合作关系和引用链条。
- **研究灵感引擎**: 允许本地大模型基于您精选的阅读列表，碰撞生成全新的研究方向与提纲。
- **写作风格润色**: 以学术母语者的专业标准，对您的学术句子和段落进行重写和润色。
- **自动化文献摘要**: 自动生成简洁的摘要卡片，大幅加快文献综述和阅读速度。

---

## 🛠️ 技术栈基石

本项目基于现代、强大的顶级开源工具构建：

*   **UI 框架**: [Streamlit](https://streamlit.io/) 提供极速的交互式网页界面
*   **向量数据库**: [ChromaDB](https://www.trychroma.com/) 驱动闪电般的语义级检索引擎
*   **LLM 编排层**: [LangChain](https://www.langchain.com/) 负责 RAG 模块与复杂文本数据流的处理
*   **本地推理引擎**: [Ollama](https://ollama.com/) 确保 LLM 调用的绝对隐私优先
*   **数据富集源**: Semantic Scholar, OpenAlex, 及 arXiv 的原生开放 API

---

## 📦 详细安装说明

### 前置要求
*   Python 3.10+
*   [Ollama](https://ollama.com) (用于本地大模型端侧推理)

### 安装步骤
1.  **克隆代码仓库**
    ```bash
    git clone https://github.com/brandonsongli-ctrl/arxiv-assistant.git
    cd arxiv-assistant
    ```

2.  **安装并运行 Ollama**
    大语言模型推理必备。
    *   在 [ollama.com](https://ollama.com) 下载并完成安装。
    *   验证安装成功：`ollama --version`
    *   下载模型：`ollama run llama3`

3.  **安装核心 Python 库**
    ```bash
    pip install -r requirements.txt
    ```

4.  **安装 Spacy 模型**
    用于基础语言结构分析：
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **配置 OCR 引擎 (可选)**
    针对无文本层的扫描版早年文献 PDF。
    *   请在您的操作系统中安装原生依赖 `poppler` 和 `tesseract`。
    *   安装配套的 Python 包：`pip install pdf2image pytesseract`
    *   并在 `.env` 文件中添加变量 `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK=1`。

---

## ⚙️ 系统配置项

通过修改环境变量或 `.env` 配置文件自定义应用行为：

| 环境变量标识 | 功能描述 | 默认配置 |
| :--- | :--- | :--- |
| `ARXIV_ASSISTANT_DB_DIR` | ChromaDB 物理持久化目录 | `./data/chroma_db` |
| `ARXIV_ASSISTANT_EMBEDDING_MODEL` | 文本切片的 Embedding 模型名称 | `all-MiniLM-L6-v2` |
| `ARXIV_ASSISTANT_RETRIEVAL_MODE` | 检索引擎模式 (vector/bm25/hybrid) | `hybrid` |
| `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK` | 开启扫描版 PDF OCR 备用机制 | `0` |

*如需查阅全量内置项，请见 `src/config.py`。*

---

## 🤝 欢迎贡献代码

如果您有更好的想法或优化，非常欢迎提交 Pull Request 和反馈！
