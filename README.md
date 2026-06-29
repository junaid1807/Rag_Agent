# Rag_Agent
A multi-tool agentic chatbot built with LlamaIndex + LangChain + Groq.
---
title: RAG Agentic Chatbot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 RAG Agentic Chatbot

A multi-tool agentic chatbot built with **LlamaIndex + LangChain + Groq**.

## Architecture

- **LlamaIndex** → Document ingestion, FAISS vector store, semantic retrieval  
- **LangChain ReAct Agent** → Tool orchestration & step-by-step reasoning  
- **LangGraph** → Agent graph execution with memory  
- **Groq (Llama-3.1 8B)** → Free, fast LLM inference  

## Tools

| Tool | Source |
|------|--------|
| 📚 RAG Knowledge Base | Local FAISS index via LlamaIndex |
| 🌐 Web Search | DuckDuckGo (no key needed) |
| 📖 Wikipedia | LangChain Wikipedia wrapper |
| 🔬 ArXiv | Academic paper search |

## Usage

1. Enter your free [Groq API key](https://console.groq.com)
2. Ask anything — the agent picks the right tool(s) automatically

## Stack

`llama-index` · `langchain` · `langgraph` · `langchain-groq` · `faiss-cpu` · `gradio`