"""knowledge_rag - a small, end-to-end Retrieval-Augmented Generation pipeline.

Index time  (python ingest.py):  loaders -> chunking -> embeddings -> vector_store (+ manifest)
Query time  (python chat.py):    chat_engine -> retriever (embeddings + bm25 -> RRF) -> prompts -> llm

Each stage is one module so it can be read, tested and swapped on its own:

    config.py        config.yaml -> typed Config (providers, keys from env, tunables)
    loaders.py       files (txt/md/pdf/docx/csv/html/json) -> Document(text, source, metadata)
    chunking.py      Document -> overlapping Chunk passages cut at paragraph/sentence boundaries
    embeddings.py    text -> unit vectors (OpenAI, Gemini, Ollama, local sentence-transformers, builtin)
    vector_store.py  persistent NumPy index + cosine search + per-source removal
    bm25.py          keyword (sparse) retrieval, implemented from the formula
    retriever.py     hybrid dense + sparse search fused with Reciprocal Rank Fusion
    prompts.py       grounding rules, numbered context, follow-up rewrite prompt
    llm.py           chat adapters (OpenAI, Anthropic, Gemini, Grok, Ollama) with one interface
    chat_engine.py   rewrite -> retrieve -> augment -> generate, with bounded conversation memory
    pipeline.py      wiring for both paths: ingest() and build_chat_engine()
"""

__version__ = "1.0.0"
