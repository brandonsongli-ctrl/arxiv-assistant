# Local Academic Assistant - 使用说明（中文）

[English](README.md) | [简体中文](README.zh-CN.md)

## 环境准备

1. **安装依赖**  
   在项目根目录打开终端并执行：
   ```bash
   pip install -r requirements.txt
   ```
   说明：
   - Google Scholar 检索需要 `scholarly`（已包含在 `requirements.txt`）。
   - 句式模式分析需要 `spacy` 和英文模型：
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - 可选聚类功能需要 `umap-learn` 和 `hdbscan`。

2. **安装并运行 Ollama（可选，本地 LLM）**  
   - 从 [ollama.com](https://ollama.com) 下载并安装。  
   - 在另一个终端执行 `ollama run llama3`（或 `mistral`）下载并启动模型。  
   - 默认监听端口为 `11434`。

## 运行应用

1. 启动 Streamlit：
   ```bash
   streamlit run src/app.py
   ```
2. 浏览器会自动打开应用页面。

## 功能概览

1. **数据库管理**：在侧边栏检索 arXiv 主题（例如 “Mechanism Design”）并下载论文。  
2. **文献语义检索**：在 “Semantic Search” 页签中按问题或概念检索。  
3. **学术句子润色**：在 “Rewrite” 页签粘贴草稿句子，系统会检索相似语料并进行改写。  
4. **研究想法生成**：输入主题后，系统基于你的文献库给出新方向建议。  
5. **批量导入**：支持多关键词检索并自动导入去重后的结果。  
6. **元数据编辑**：可在 UI 直接修改标题/作者/年份/DOI。  
7. **丰富导出**：支持导出 BibTeX、JSON、CSL-JSON、RIS、Zotero JSON。  
8. **论文摘要卡片**：可生成论文级摘要。  
9. **综述参考文献**：文献综述支持参考文献清单。  
10. **风格化改写**：可切换 journal / working paper / grant 风格。  
11. **批量下载**：可勾选多条检索结果一键导入。  
12. **结构化想法输出**：研究想法支持结构化格式。  
13. **搜索历史**：保存并快速复用近期搜索。  
14. **快速筛选**：按年份、来源、作者过滤文献库。  
15. **增量导入**：扫描 `data/pdfs` 并仅导入新增文件。  
16. **诊断工具**：启动时检查 DB/LLM/依赖状态。  
17. **检索参数调优**：运行时调节检索模式、alpha、reranker。  
18. **检索体验升级**：查询扩展、关键词高亮、证据片段长度调节。  
19. **元数据覆盖增强**：venue 在检索、富化、导入和导出链路中贯通。  
20. **知识管理**：支持 tags/favorites/reading list 与快速筛选、行内操作。  
21. **后台任务队列**：异步下载 -> 导入 -> 富化，支持取消/重试/恢复。  
22. **冲突合并 UI**：可视化重复组审核与元数据保留式合并。  
23. **证据化回答**：生成带引用标签（`[S1]`, `[S2]`）与置信度的回答。  
24. **图谱分析**：支持 citation / co-citation / coupling / author collaboration。  
25. **Watchlist 与 Digest**：支持按关键词/作者的日/周监控与摘要文件。  
26. **OCR 回退**：对低文本 PDF 可选 OCR 提取。

## 配置项

可通过环境变量覆盖默认配置：

- `ARXIV_ASSISTANT_DB_DIR`：自定义 ChromaDB 存储路径  
- `ARXIV_ASSISTANT_EMBEDDING_MODEL`：`sentence-transformers` 模型名  
- `ARXIV_ASSISTANT_CHUNK_SIZE`：分块大小（字符）  
- `ARXIV_ASSISTANT_CHUNK_OVERLAP`：分块重叠（字符）  
- `ARXIV_ASSISTANT_RETRIEVAL_MODE`：`vector`、`bm25`、`hybrid`（默认）  
- `ARXIV_ASSISTANT_HYBRID_ALPHA`：混合检索中向量分数权重（0-1）  
- `ARXIV_ASSISTANT_RERANKER_MODEL`：可选 CrossEncoder 重排模型名  
- `ARXIV_ASSISTANT_RERANKER_TOP_K`：重排候选 Top-K  
- `ARXIV_ASSISTANT_HYBRID_CANDIDATE_MULTIPLIER`：混合检索候选扩展倍数  
- `ARXIV_ASSISTANT_HISTORY_PATH`：搜索历史文件路径  
- `ARXIV_ASSISTANT_TASK_QUEUE_PATH`：后台任务队列 JSON 路径  
- `ARXIV_ASSISTANT_WATCHLIST_PATH`：watchlist JSON 路径  
- `ARXIV_ASSISTANT_DIGEST_DIR`：digest 输出目录  
- `ARXIV_ASSISTANT_ENABLE_OCR_FALLBACK`：`1`/`0` 开关 OCR 回退  
- `ARXIV_ASSISTANT_OCR_MAX_PAGES`：触发 OCR 时最多处理前 N 页  
- `ARXIV_ASSISTANT_OCR_MIN_TEXT_CHARS`：触发 OCR 的最小文本阈值  
- `SCHOLAR_PROXY`：Google Scholar 单代理 URL（例如 `http://user:pass@host:port`）  
- `SCHOLAR_USE_FREE_PROXY`：设为 `1` 尝试免费代理  
- `SCHOLAR_USE_TOR`：设为 `1` 尝试 Tor 代理

### OCR 可选依赖

- Python 包：`pdf2image`、`pytesseract`  
- 系统工具：`poppler`、`tesseract`

### Docker Compose 说明

- `docker-compose.yml` 会将 ChromaDB 持久化到 `/app/data/chroma_db`，搜索历史持久化到 `/app/data/history/search_history.json`。

