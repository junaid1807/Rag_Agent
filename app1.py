import os
import uuid
import gradio as gr
import faiss

from dotenv import load_dotenv
load_dotenv()

# ── LlamaIndex ────────────────────────────────────────────────────────────────
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.file import PDFReader, DocxReader

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

import arxiv as arxiv_lib

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
LLM_MODEL     = "llama-3.1-8b-instant"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
EMBED_DIM     = 384
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5

# ═════════════════════════════════════════════════════════════════════════════
# LLAMAINDEX GLOBAL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ═════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (module-level, shared & mutable)
# ═════════════════════════════════════════════════════════════════════════════
SAMPLE_TEXTS = [
    """Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of
    large language models with external knowledge retrieval. In RAG, relevant documents are
    fetched from a vector store based on the user query, and then passed as context to the LLM
    to generate a grounded, accurate response. This reduces hallucinations and keeps information
    up-to-date without expensive retraining.""",

    """LlamaIndex (formerly GPT Index) is a data framework for building LLM-powered applications.
    It provides connectors to ingest data from various sources (PDFs, APIs, databases),
    efficient indexing using vector stores like FAISS or Pinecone, and query engines with
    sub-question decomposition, re-ranking, and hybrid search. LlamaIndex integrates seamlessly
    with LangChain for agentic workflows.""",

    """LangChain is a framework for developing applications powered by language models.
    It enables building agents that can reason step-by-step using the ReAct (Reasoning + Acting)
    paradigm. Agents decide which tools to call — such as search, calculators, or databases —
    based on the user's goal, execute actions, observe results, and iterate until a final
    answer is reached.""",

    """FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
    and clustering of dense vectors. It supports billion-scale vector search and provides
    multiple index types: Flat (exact), IVF (approximate), and HNSW (graph-based).
    FAISS is highly optimized for GPU and CPU and is widely used as the backbone
    for vector databases in RAG systems.""",

    """Agentic AI refers to systems where an LLM autonomously plans and executes multi-step
    tasks by calling tools, APIs, or sub-agents. Unlike a single-turn chatbot, an agentic
    system maintains state across steps, backtracks on failure, and selects the best
    strategy from multiple options. Key components include a planner, a tool executor,
    memory, and an evaluator.""",
]

_faiss_index  = faiss.IndexFlatL2(EMBED_DIM)
_vector_store = FaissVectorStore(faiss_index=_faiss_index)
_storage_ctx  = StorageContext.from_defaults(vector_store=_vector_store)
_index        = VectorStoreIndex.from_documents(
    [Document(text=t) for t in SAMPLE_TEXTS],
    storage_context=_storage_ctx,
)

# track uploaded doc names for the UI
_doc_names: list[str] = ["[built-in] RAG overview (5 sample docs)"]


def get_retriever():
    return _index.as_retriever(similarity_top_k=TOP_K)


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _ingest_documents(docs: list[Document], source_name: str) -> str:
    """Insert docs into the live FAISS index and return a status message."""
    for doc in docs:
        _index.insert(doc)
    _doc_names.append(source_name)
    return f"✅ Indexed **{source_name}** — {len(docs)} chunk(s) added. Total vectors: {_faiss_index.ntotal}"


def upload_file(files) -> tuple[str, str]:
    """
    Gradio callback: accepts one or more uploaded files.
    Supports .txt, .pdf, .docx, .md
    Returns (status_message, updated doc list markdown).
    """
    if not files:
        return "⚠️ No file received.", _doc_list_md()

    messages = []
    for file_obj in files:
        path = file_obj.name  # temp path on disk
        fname = os.path.basename(path)
        ext = os.path.splitext(fname)[1].lower()

        try:
            if ext in (".txt", ".md"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if not text:
                    messages.append(f"⚠️ `{fname}` is empty.")
                    continue
                docs = [Document(text=text, metadata={"source": fname})]

            elif ext == ".pdf":
                reader = PDFReader()
                docs = reader.load_data(file=path)
                for d in docs:
                    d.metadata["source"] = fname

            elif ext == ".docx":
                reader = DocxReader()
                docs = reader.load_data(file=path)
                for d in docs:
                    d.metadata["source"] = fname

            else:
                messages.append(f"❌ `{fname}`: unsupported format (use .txt, .md, .pdf, .docx).")
                continue

            msg = _ingest_documents(docs, fname)
            messages.append(msg)

        except Exception as e:
            messages.append(f"❌ `{fname}`: {str(e)}")

    return "\n\n".join(messages), _doc_list_md()


def add_text_snippet(text: str) -> tuple[str, str]:
    """Gradio callback: paste raw text directly into the index."""
    text = text.strip()
    if not text:
        return "⚠️ Please enter some text first.", _doc_list_md()
    label = f"[pasted text] {text[:40]}..."
    docs  = [Document(text=text, metadata={"source": "pasted"})]
    msg   = _ingest_documents(docs, label)
    return msg, _doc_list_md()


def _doc_list_md() -> str:
    lines = [f"- {name}" for name in _doc_names]
    return "**Indexed sources:**\n" + "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# LANGCHAIN TOOLS
# ═════════════════════════════════════════════════════════════════════════════
@tool
def rag_knowledge_base(query: str) -> str:
    """Search the local knowledge base using semantic similarity.
    Use this FIRST for any question about RAG, LlamaIndex, LangChain, FAISS,
    agentic AI, or any uploaded documents."""
    nodes = get_retriever().retrieve(query)
    if not nodes:
        return "No relevant documents found in the knowledge base."
    return "\n\n".join(
        f"[Chunk {i} | Score: {round(n.score, 4) if n.score else 'N/A'}]\n{n.get_content().strip()}"
        for i, n in enumerate(nodes, 1)
    )

@tool
def web_search(query: str) -> str:
    """Search the web for current events or information not in the knowledge base."""
    return DuckDuckGoSearchRun().run(query)

@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for factual and encyclopedic information."""
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    ).run(query)

@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for academic research papers."""
    try:
        client  = arxiv_lib.Client(page_size=3, delay_seconds=5, num_retries=2)
        results = list(client.results(
            arxiv_lib.Search(query=query, max_results=3, sort_by=arxiv_lib.SortCriterion.Relevance)
        ))
        if not results:
            return "No papers found."
        return "\n\n---\n\n".join(
            f"**{r.title}**\nAuthors: {', '.join(a.name for a in r.authors[:3])}\n"
            f"Published: {r.published.date()}\nSummary: {r.summary[:300]}...\nURL: {r.entry_id}"
            for r in results
        )
    except Exception as e:
        return f"ArXiv temporarily unavailable. Error: {str(e)}"

TOOLS = [rag_knowledge_base, web_search, wikipedia, arxiv_search]

# ═════════════════════════════════════════════════════════════════════════════
# AGENT BUILDER
# ═════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a helpful research assistant with access to four tools.

INSTRUCTIONS:
1. Always call rag_knowledge_base FIRST for any question — it may contain user-uploaded documents.
2. Read the tool output carefully.
3. Write a complete, detailed answer based on what the tool returned.
4. End your response by mentioning which tool(s) you used.
5. If a tool returns an error or rate limit message, use other tools or answer from your knowledge.

IMPORTANT: Never stop after calling a tool. Always write a full answer using the tool results."""

def build_agent(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key
    Settings.llm = LlamaGroq(model=LLM_MODEL, temperature=0)
    llm = ChatGroq(
        model=LLM_MODEL, temperature=0, streaming=False,
        model_kwargs={"parallel_tool_calls": False}
    )
    return create_react_agent(
        model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver()
    )

# ═════════════════════════════════════════════════════════════════════════════
# CHAT HANDLER
# ═════════════════════════════════════════════════════════════════════════════
def respond(message, history, api_key):
    if not api_key.strip():
        yield "⚠️ Please enter your Groq API key in the sidebar."
        return
    try:
        agent  = build_agent(api_key.strip())
        result = agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        answer = ""
        for msg in reversed(result["messages"]):
            if (isinstance(msg, AIMessage)
                    and isinstance(msg.content, str)
                    and msg.content.strip()
                    and not msg.tool_calls):
                answer = msg.content
                break
        yield answer or "⚠️ Agent finished but produced no text answer."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Agentic Chatbot") as demo:

    gr.Markdown("""
# 🤖 RAG Agentic Chatbot
**LlamaIndex** · **LangChain ReAct Agent** · **Groq Llama-3** · **FAISS**
""")

    with gr.Row():

        # ── LEFT SIDEBAR ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            api_key = gr.Textbox(
                label="🔑 Groq API Key",
                placeholder="gsk_...",
                type="password",
                info="Free key at console.groq.com"
            )

            gr.Markdown("---")
            gr.Markdown("### 📂 Add Documents to RAG")

            # File upload
            file_upload = gr.File(
                label="Upload file(s)",
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf", ".docx"],
            )
            upload_btn    = gr.Button("⬆️ Index uploaded file(s)", variant="primary")
            upload_status = gr.Markdown("")

            gr.Markdown("**— or paste text —**")

            # Text paste
            paste_box  = gr.Textbox(
                label="Paste text snippet",
                placeholder="Paste any text here to add it to the knowledge base...",
                lines=4,
            )
            paste_btn    = gr.Button("➕ Add pasted text")
            paste_status = gr.Markdown("")

            gr.Markdown("---")
            doc_list = gr.Markdown(_doc_list_md())

            gr.Markdown("""
---
**Stack**
`LlamaIndex` · `LangChain` · `LangGraph`
`Groq` · `FAISS` · `HuggingFace Embeddings`

**Tools**
📚 RAG · 🌐 Web · 📖 Wiki · 🔬 ArXiv
""")

        # ── CHAT PANEL ────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[api_key],
                examples=[
                    "What is RAG and how does LlamaIndex enable it?",
                    "Find recent ArXiv papers on LLM agents",
                    "Explain FAISS vector similarity search",
                    "What is the ReAct prompting framework?",
                    "What are the latest AI agent developments in 2025?",
                ],
                chatbot=gr.Chatbot(height=520, show_label=False),
                textbox=gr.Textbox(placeholder="Ask anything...", scale=7),
            )

    # ── BUTTON CALLBACKS ──────────────────────────────────────────────────────
    upload_btn.click(
        fn=upload_file,
        inputs=[file_upload],
        outputs=[upload_status, doc_list],
    )
    paste_btn.click(
        fn=add_text_snippet,
        inputs=[paste_box],
        outputs=[paste_status, doc_list],
    )

demo.launch()
