import os
import re
import shutil
import tempfile
import traceback

import streamlit as st
import fitz  # PyMuPDF

from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ENVIRONMENT VARIABLES
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "./data/researchlens_qdrant"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "researchlens_papers"
)

# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# 4. SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "paper_names" not in st.session_state:
    st.session_state.paper_names = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

# 5. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 ResearchLens AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered research paper reading, retrieval and analysis assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# 6. EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

# 7. QDRANT CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    os.makedirs(QDRANT_PATH, exist_ok=True)

    return QdrantClient(
        path=QDRANT_PATH
    )

# 8. OLLAMA LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_llm(model_name):

    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ ResearchLens Settings")

    st.subheader("🤖 Local AI Model")

    model_name = st.text_input(
        "Ollama Model",
        value=OLLAMA_MODEL,
        help="Example: llama3.1:8b or qwen2.5:3b"
    )

    st.caption(
        f"Ollama Server: {OLLAMA_BASE_URL}"
    )

    st.divider()

    st.subheader("📄 Document Processing")

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=400,
        value=150,
        step=50
    )

    st.divider()

    st.subheader("🔎 Retrieval")

    vector_top_k = st.slider(
        "Vector results",
        min_value=2,
        max_value=10,
        value=5
    )

    bm25_top_k = st.slider(
        "Keyword results",
        min_value=2,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.write(
        f"**Indexed papers:** "
        f"{len(st.session_state.paper_names)}"
    )

    st.write(
        f"**Indexed chunks:** "
        f"{st.session_state.document_count}"
    )

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        try:

            client = get_qdrant_client()

            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(
                    collection_name=COLLECTION_NAME
                )

        except Exception:
            pass

        st.session_state.vector_store = None
        st.session_state.paper_names = []
        st.session_state.chat_history = []
        st.session_state.documents = []
        st.session_state.bm25 = None
        st.session_state.document_count = 0

        st.success("Knowledge base cleared.")

        st.rerun()

    st.divider()

    st.caption(
        "ResearchLens AI uses Retrieval-Augmented Generation "
        "(RAG) with Ollama, Qdrant, Hugging Face embeddings "
        "and BM25 keyword retrieval."
    )
# 10. PDF PROCESSING
# ============================================================

def process_pdfs(
    uploaded_files,
    chunk_size,
    chunk_overlap
):

    documents = []
    processed_names = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        status_text.write(
            f"Processing: {uploaded_file.name}"
        )

        try:

            pdf_bytes = uploaded_file.getvalue()

            pdf_document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            for page_number, page in enumerate(pdf_document):

                text = page.get_text("text")

                if not text.strip():
                    continue

                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": uploaded_file.name,
                            "page": page_number + 1
                        }
                    }
                )

            pdf_document.close()

            processed_names.append(
                uploaded_file.name
            )

        except Exception as error:

            st.warning(
                f"Could not process "
                f"{uploaded_file.name}: {error}"
            )

        progress_bar.progress(
            (index + 1) / len(uploaded_files)
        )

    progress_bar.empty()
    status_text.empty()

    if not documents:

        raise ValueError(
            "No readable text was extracted "
            "from the uploaded PDFs."
        )

    # Convert extracted pages into LangChain documents
    # --------------------------------------------------------

    from langchain_core.documents import Document

    langchain_documents = []

    for item in documents:

        langchain_documents.append(
            Document(
                page_content=item["text"],
                metadata=item["metadata"]
            )
        )

    # Split into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    chunks = splitter.split_documents(
        langchain_documents
    )

    if not chunks:

        raise ValueError(
            "No text chunks were generated."
        )

    # Add unique chunk IDs

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    return chunks, processed_names

# 11. CREATE QDRANT VECTOR STORE
# ============================================================

def create_vector_store(chunks):
    embeddings = load_embedding_model()
    client = get_qdrant_client()

    # Delete the old collection if it exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Generate embeddings to determine the vector dimension
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)

    # Create the Qdrant collection
    client.create_collection(
     collection_name=COLLECTION_NAME,
     vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)

    # Initialize the vector store using the existing client
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # Store the document chunks
    vector_store.add_documents(chunks)

    return vector_store

# 12. CREATE BM25 INDEX
# ============================================================

def create_bm25_index(chunks):

    tokenized_documents = []

    for document in chunks:

        tokens = re.findall(
            r"\b\w+\b",
            document.page_content.lower()
        )

        tokenized_documents.append(tokens)

    if not tokenized_documents:
        return None

    return BM25Okapi(
        tokenized_documents
    )

# 13. BM25 RETRIEVAL
# ============================================================

def bm25_retrieve(
    question,
    documents,
    bm25,
    top_k=5
):

    if not documents or bm25 is None:
        return []

    query_tokens = re.findall(
        r"\b\w+\b",
        question.lower()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:top_k]:

        if scores[index] > 0:

            results.append(
                documents[index]
            )

    return results


# 14. HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    question,
    vector_store,
    documents,
    bm25,
    vector_top_k,
    bm25_top_k
):

    retrieved_documents = []

    # Vector retrieval
    # --------------------------------------------------------

    if vector_store is not None:

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": vector_top_k
            }
        )

        vector_documents = retriever.invoke(
            question
        )

        retrieved_documents.extend(
            vector_documents
        )

    # BM25 keyword retrieval
    # --------------------------------------------------------

    keyword_documents = bm25_retrieve(
        question,
        documents,
        bm25,
        bm25_top_k
    )

    retrieved_documents.extend(
        keyword_documents
    )

    # Remove duplicate chunks
    # --------------------------------------------------------

    unique_documents = {}

    for document in retrieved_documents:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id is None:

            chunk_id = (
                document.metadata.get("source", "")
                + str(document.metadata.get("page", ""))
                + document.page_content[:100]
            )

        unique_documents[
            str(chunk_id)
        ] = document

    return list(
        unique_documents.values()
    )

# 15. CREATE CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page"
        )

        if page:

            location = (
                f"{source}, page {page}"
            )

        else:

            location = source

        context_parts.append(
            f"Source: {location}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )

# 16. ANSWER QUESTION
# ============================================================

def answer_question(
    question,
    vector_store,
    documents,
    bm25,
    model_name,
    vector_top_k,
    bm25_top_k
):

    if vector_store is None:

        return (
            "Please upload and process "
            "research papers first.",
            []
        )

   
    # Load Ollama
    # --------------------------------------------------------

    llm = load_llm(
        model_name
    )

    # Hybrid retrieval
    # --------------------------------------------------------

    relevant_documents = hybrid_retrieve(
        question=question,
        vector_store=vector_store,
        documents=documents,
        bm25=bm25,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k
    )

    if not relevant_documents:

        return (
            "I could not find relevant information "
            "in the uploaded research papers.",
            []
        )

    # Build context
    # --------------------------------------------------------

    context = build_context(
        relevant_documents
    )

    # ResearchLens prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are ResearchLens AI, an AI research
paper analysis assistant.

Your task is to answer questions using
ONLY the retrieved research-paper context.

Rules:

1. Use the supplied context as your primary evidence.

2. Do not invent facts, statistics, authors,
   results, references or citations.

3. If the answer cannot be found in the
   supplied context, clearly say:

   "The uploaded research papers do not
   provide enough information to answer
   this question."

4. Explain technical concepts in simple
   and clear language when appropriate.

5. Use headings, bullet points and numbered
   lists when useful.

6. Mention the research paper filename
   and page number when available.

7. Clearly distinguish between:
   - What the paper says
   - Your explanation of the paper

8. Do not use outside knowledge to fill
   missing information.

Retrieved research-paper context:

{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    # RAG chain
    # --------------------------------------------------------

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return (
        answer,
        relevant_documents
    )

# 17. MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Upload Papers",
        "💬 Ask ResearchLens",
        "📝 Paper Summary",
        "📊 Research Insights"
    ]
)


# ============================================================
# 18. TAB 1 - UPLOAD PAPERS
# ============================================================

with tab1:

    st.header(
        "📄 Upload Research Papers"
    )

    st.write(
        "Upload one or more research papers "
        "to build your ResearchLens knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based research papers."
    )

    if uploaded_files:

        st.subheader(
            "Selected Research Papers"
        )

        for file in uploaded_files:

            st.write(
                f"📄 **{file.name}** "
                f"({file.size / 1024:.1f} KB)"
            )

        if st.button(
            "🚀 Process Research Papers",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Extracting text and building "
                    "the research knowledge base..."
                ):

                    chunks, paper_names = process_pdfs(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap
                    )

                    # Create Qdrant database

                    vector_store = create_vector_store(
                        chunks
                    )

                    # Create BM25 index

                    bm25 = create_bm25_index(
                        chunks
                    )

                    # Save everything

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.documents = (
                        chunks
                    )

                    st.session_state.bm25 = bm25

                    st.session_state.paper_names = (
                        paper_names
                    )

                    st.session_state.document_count = (
                        len(chunks)
                    )

                    st.session_state.chat_history = []

                st.success(
                    f"Successfully indexed "
                    f"{len(paper_names)} research paper(s) "
                    f"into {len(chunks)} chunks."
                )

            except Exception as e:
              st.error(f"Error while processing papers: {e}")
              st.code(traceback.format_exc())

# 19. TAB 2 - ASK RESEARCHLENS
# ============================================================

with tab2:

    st.header(
        "💬 Ask ResearchLens"
    )

    if not st.session_state.vector_store:

        st.info(
            "📄 Upload and process research papers "
            "before asking questions."
        )

    else:

        question = st.chat_input(
            "Ask a question about your research papers..."
        )

        # Display previous conversation

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        if question:

            # User message

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            # AI response

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching research papers..."
                ):

                    try:

                        answer, sources = answer_question(
                            question=question,
                            vector_store=(
                                st.session_state.vector_store
                            ),
                            documents=(
                                st.session_state.documents
                            ),
                            bm25=(
                                st.session_state.bm25
                            ),
                            model_name=model_name,
                            vector_top_k=vector_top_k,
                            bm25_top_k=bm25_top_k
                        )

                        st.markdown(answer)

                        # Sources

                        if sources:

                            with st.expander(
                                "📚 Retrieved Sources"
                            ):

                                shown_sources = set()

                                for source in sources:

                                    filename = source.metadata.get(
                                        "source",
                                        "Unknown"
                                    )

                                    page = source.metadata.get(
                                        "page",
                                        "Unknown"
                                    )

                                    source_key = (
                                        filename,
                                        page
                                    )

                                    if source_key in shown_sources:
                                        continue

                                    shown_sources.add(
                                        source_key
                                    )

                                    st.write(
                                        f"📄 **{filename}** "
                                        f"— Page {page}"
                                    )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    except Exception as error:

                        st.error(
                            f"Error communicating with "
                            f"Ollama: {error}"
                        )

# 20. TAB 3 - PAPER SUMMARY
# ============================================================

with tab3:

    st.header(
        "📝 Research Paper Summary"
    )

    if not st.session_state.documents:

        st.info(
            "Upload and process a research paper first."
        )

    else:

        selected_paper = st.selectbox(
            "Select paper",
            st.session_state.paper_names
        )

        if st.button(
            "✨ Generate Summary",
            type="primary"
        ):

            paper_documents = [
                document
                for document
                in st.session_state.documents
                if document.metadata.get(
                    "source"
                ) == selected_paper
            ]

            if not paper_documents:

                st.warning(
                    "No content found for this paper."
                )

            else:

                paper_text = "\n\n".join(
                    document.page_content
                    for document in paper_documents
                )

                # Prevent excessively large prompts

                paper_text = paper_text[:30000]

                summary_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
You are ResearchLens AI.

Summarize the supplied research paper.

Include:

1. Research problem
2. Objectives
3. Methodology
4. Dataset or experimental setup
5. Main findings
6. Results
7. Limitations
8. Conclusion

Use ONLY the supplied paper content.
Do not invent information.

Paper content:

{paper}
"""
                        ),
                        (
                            "human",
                            "Generate a structured summary."
                        )
                    ]
                )

                try:

                    llm = load_llm(
                        model_name
                    )

                    summary_chain = (
                        summary_prompt
                        | llm
                        | StrOutputParser()
                    )

                    with st.spinner(
                        "Analyzing the research paper..."
                    ):

                        summary = summary_chain.invoke(
                            {
                                "paper": paper_text
                            }
                        )

                    st.markdown(summary)

                except Exception as error:

                    st.error(
                        f"Summary generation failed: "
                        f"{error}"
                    )


# 21. TAB 4 - RESEARCH INSIGHTS
# ============================================================

with tab4:

    st.header(
        "📊 Research Insights"
    )

    if not st.session_state.documents:

        st.info(
            "Upload research papers to see insights."
        )

    else:

        total_papers = len(
            st.session_state.paper_names
        )

        total_chunks = (
            st.session_state.document_count
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Research Papers",
                total_papers
            )

        with col2:

            st.metric(
                "Text Chunks",
                total_chunks
            )

        with col3:

            st.metric(
                "Retrieval Method",
                "Hybrid"
            )

        st.divider()

        st.subheader(
            "📚 Indexed Research Papers"
        )

        for paper in st.session_state.paper_names:

            paper_chunks = [
                document
                for document
                in st.session_state.documents
                if document.metadata.get(
                    "source"
                ) == paper
            ]

            st.write(
                f"📄 **{paper}** — "
                f"{len(paper_chunks)} chunks"
            )

        st.divider()

# 22. FOOTER
# ============================================================

st.divider()

st.caption(
    "ResearchLens AI • RAG-based Research Paper "
    "Analysis using Ollama + Qdrant + Hugging Face + BM25"
)
