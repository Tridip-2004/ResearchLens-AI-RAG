# ResearchLens AI

**An AI-powered research paper reading, retrieval, summarization, and
analysis assistant.**

ResearchLens AI helps users explore research papers by uploading
documents, asking questions about their content, generating paper
summaries, and reviewing research insights. It is designed around a
Retrieval-Augmented Generation (RAG) workflow and can use a locally
running Ollama language model to generate responses grounded in
available paper content.

<img width="1920" height="1080" alt="Screenshot 2026-11-27 012322" src="https://github.com/user-attachments/assets/93ae750b-b47e-49b9-b03b-c4499c3f10b3" />

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [How It Works](#how-it-works)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Configuration](#configuration)
-   [Run the Application](#run-the-application)
-   [Using ResearchLens AI](#using-researchlens-ai)
-   [RAG Workflow](#rag-workflow)
-   [Troubleshooting](#troubleshooting)
-   [Limitations and Responsible Use](#limitations-and-responsible-use)
-   [Future Improvements](#future-improvements)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

ResearchLens AI is a research-assistance application intended to make
academic papers easier to explore. Rather than relying only on a
language model's general knowledge, its RAG workflow retrieves relevant
passages from processed papers and supplies them as context for a
response.

The project includes a Streamlit user interface and a FastAPI backend.
The Streamlit interface provides paper-uploading and research
interaction screens. The FastAPI service exposes backend functionality
over HTTP. Depending on configuration, the applications may use separate
data stores; configure their database paths deliberately if they need to
access the same documents.

## Key Features

-   **Research paper upload:** Upload research documents for processing.
-   **Ask ResearchLens:** Ask questions about available paper content.
-   **Paper summaries:** Generate summaries using the configured
    language model and document context.
-   **Research insights:** Explore information supported by processed
    research material.
-   **RAG-based answers:** Retrieve relevant passages before generating
    a response.
-   **Local language model support:** Connect to Ollama to run a model
    locally.
-   **Vector search:** Store and retrieve document embeddings with
    Qdrant.
-   **Configurable chunking:** Adjust chunk size and overlap to control
    document segmentation.
-   **Retrieval enhancements:** The Streamlit implementation includes
    BM25 keyword-retrieval components; the FastAPI retrieval module is
    configured for hybrid retrieval and reranking.
-   **FastAPI endpoints:** Run the backend separately and inspect its
    API documentation.

Feature availability can differ between Streamlit and FastAPI. Do not
assume both interfaces share uploaded documents unless they are
configured to use a common supported storage service.

## How It Works

1.  Upload a supported research document.
2.  Extract text from the document.
3.  Split the text into overlapping chunks.
4.  Generate embeddings for the chunks.
5.  Store vectors and document information in Qdrant.
6.  Submit a question or summary request.
7.  Retrieve relevant passages from indexed content.
8.  Generate a response using the configured language model and
    retrieved context.
9.  Display the result in the ResearchLens interface.

Response quality depends on document extraction, chunking, retrieval,
model configuration, and prompt design.

## Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Interactive frontend    Streamlit               Research-paper
                                                  interface

  Backend API             FastAPI                 HTTP API and backend
                                                  functionality

  Language-model runtime  Ollama                  Runs a locally
                                                  available model

  Language model          Configurable Ollama     Generates answers and
                          model,                  summaries
                          e.g. `qwen2.5:3b`       

  Vector database         Qdrant                  Stores and retrieves
                                                  embeddings

  Embeddings              Sentence Transformers / Converts text into
                          Hugging Face embeddings vectors

  PDF processing          PyMuPDF (`fitz`)        Extracts PDF text

  Text chunking           LangChain text          Divides text into
                          splitters               chunks

  Retrieval               Vector retrieval and    Finds relevant passages
                          BM25 components         

  Reranking               Cross-encoder model in  Refines retrieval
                          backend configuration   ranking

  Configuration           `.env` and application  Controls paths, models,
                          settings                and options
  -----------------------------------------------------------------------

Exact package versions depend on the project's environment and
dependency files.

## Project Structure

The following is a simplified representation of the current development
layout:

``` text
FastAPI/
├── app/
│   ├── main.py             # FastAPI application and endpoints
│   ├── config.py           # Backend configuration
│   ├── store.py            # Qdrant storage and embeddings
│   ├── retrieval.py        # Retrieval and ranking
│   └── answer.py           # Answer generation
├── data/
│   ├── uploads/            # Uploaded documents, if configured here
│   └── qdrant/             # FastAPI local Qdrant database
├── ResearchLens-AI-RAG/
│   ├── streamlit_app.py    # Streamlit interface and RAG workflow
│   └── .streamlit/         # Optional Streamlit configuration/secrets
├── .env                    # Local environment configuration
└── README.md
```

This is a representative layout; some directories may not exist in every
checkout. Keep local databases, uploaded files, and credentials out of
version control.

## Prerequisites

-   Python version compatible with the project's dependencies.
-   Ollama installed and running.
-   At least one Ollama model downloaded locally.
-   Project Python dependencies installed.
-   Sufficient disk space for models, uploaded documents, embeddings,
    and local vector databases.
-   Git, if cloning the project from a repository.

Check Ollama from PowerShell:

``` powershell
ollama --version
ollama list
```

For example, install the configured model with:

``` powershell
ollama pull qwen2.5:3b
```

The model name in the application must match a model shown by
`ollama list`.

## Installation

### 1. Open the project directory

``` powershell
cd C:\Users\tridi\OneDrive\Desktop\FastAPI
```

If the project is located elsewhere, use that path instead.

### 2. Create and activate a virtual environment

If you do not already have one:

``` powershell
python -m venv .venv
```

Activate it:

``` powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` exists in the project root:

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the Streamlit application has its own dependency file, install the
packages listed there too. If there is no dependency file, document the
required packages in one so setup can be reproduced reliably.

### 4. Start Ollama

Make sure the Ollama service is running. Its default local API address
is:

``` text
http://localhost:11434
```

Check the installed models with `ollama list`.

## Configuration

A root `.env` file can provide local configuration. Example values for
the Streamlit application:

``` dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
QDRANT_COLLECTION=researchlens_papers
QDRANT_PATH=./data/researchlens_qdrant
```

The FastAPI backend has its own settings in `app/config.py`, including
its collection name, model, data directory, and retrieval parameters.
Check both configurations when running both applications.

### Streamlit secrets

A `secrets.toml` file is not required if the application reads settings
with `os.getenv()` and `python-dotenv`. If the code directly accesses
`st.secrets`, Streamlit may require a secrets file.

If you intentionally use Streamlit secrets, create:

``` text
ResearchLens-AI-RAG/.streamlit/secrets.toml
```

Example:

``` toml
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
QDRANT_COLLECTION = "researchlens_papers"
QDRANT_PATH = "./data/researchlens_qdrant"
```

Do not commit API keys, passwords, or other private credentials. A local
Ollama setup generally does not require an API key unless another
service in the application needs one.

### Important: Qdrant storage paths

Qdrant local storage paths must not be concurrently opened by
independent local Qdrant clients. Doing so can lead to SQLite
transaction or locking errors.

The current development setup uses `data/qdrant` for FastAPI and a
configurable `QDRANT_PATH` for Streamlit. If the applications use
different local paths, their indexed documents are separate. If both
must share the same collection and documents, configure them to use a
supported shared Qdrant server instead of pointing multiple local
clients at the same local database directory.

Stop both applications before changing paths. Do not delete a Qdrant
directory unless you are certain you no longer need its indexed data.

## Run the Application

Use separate PowerShell terminals for the backend and frontend.

### Run FastAPI

From the project root, activate the virtual environment and run:

``` powershell
python -m uvicorn app.main:app --reload
```

Default local addresses:

-   API: `http://127.0.0.1:8000`
-   Interactive API documentation: `http://127.0.0.1:8000/docs`
-   Health endpoint, if implemented: `http://127.0.0.1:8000/health`

The `--reload` option is for development, not production deployment.

### Run Streamlit

In another terminal:

``` powershell
cd C:\Users\tridi\OneDrive\Desktop\FastAPI
.\.venv\Scripts\Activate.ps1
python -m streamlit run .\ResearchLens-AI-RAG\streamlit_app.py
```

Streamlit normally opens at:

``` text
http://localhost:8501
```

Press `Ctrl+C` in the relevant terminal to stop an application.

## Using ResearchLens AI

1.  Start Ollama and verify that the configured model is installed.
2.  Launch Streamlit.
3.  Open **Upload Papers** and upload a supported research paper.
4.  Wait for text extraction and indexing to finish.
5.  Open **Ask ResearchLens** and ask a question about the uploaded
    content.
6.  Review the answer and compare it with the original paper where
    possible.
7.  Use **Paper Summary** to request a summary.
8.  Open **Research Insights** to explore insights supported by the
    current implementation.
9.  If retrieval results are poor, review extraction, chunking,
    embeddings, and retrieval settings.

Example questions:

-   What problem does this paper address?
-   What methodology did the authors use?
-   Which dataset was used?
-   What evaluation metrics are reported?
-   What are the main findings and limitations?
-   What future work do the authors suggest?

Verify important claims, numerical results, and citations against the
original paper.

## RAG Workflow

### Document ingestion

Text is extracted from supported documents and split into overlapping
chunks.

### Embedding and indexing

An embedding model converts each chunk into a vector representation.
Qdrant stores vectors and associated document information.

### Retrieval

When a question is submitted, the application searches for relevant
chunks. Depending on the code path, retrieval may combine semantic
vector search with BM25 keyword retrieval. The FastAPI backend also has
a configured cross-encoder reranker.

### Response generation

Retrieved passages are supplied as context to the configured language
model, which generates a response based on the prompt and context.

### Summary and insights

Summary and research-insight features use the document-processing and
language-model workflow implemented in the application. Accuracy depends
on the availability of relevant information in the extracted text and
supplied context.

## Configuration Options

  ------------------------------------------------------------------------------------------
  Setting                 Purpose                 Example
  ----------------------- ----------------------- ------------------------------------------
  `OLLAMA_BASE_URL`       Ollama service address  `http://localhost:11434`

  `OLLAMA_MODEL`          Streamlit model         `qwen2.5:3b`

  `QDRANT_PATH`           Streamlit local Qdrant  `./data/researchlens_qdrant`
                          directory               

  `QDRANT_COLLECTION`     Streamlit collection    `researchlens_papers`
                          name                    

  `collection_name`       FastAPI collection      `researchlens`
                          setting                 

  `embedding_model`       FastAPI embedding model `sentence-transformers/all-MiniLM-L6-v2`

  `reranker_model`        FastAPI reranker        `cross-encoder/ms-marco-MiniLM-L-6-v2`

  `top_k`                 Number of top backend   `5`
                          retrieval results       

  `chunk_size`            Backend chunk size      `1000`

  `chunk_overlap`         Backend chunk overlap   `200`
  ------------------------------------------------------------------------------------------

Streamlit may have separate chunk-size and overlap controls. Changing
one application's settings does not automatically change the other.

## Troubleshooting

### `StreamlitSecretNotFoundError: No secrets found`

The application accesses `st.secrets`, but Streamlit cannot find a
secrets file. If secrets are unnecessary, read settings through
`os.getenv()` and load `.env` with `load_dotenv()`. Otherwise, create
`.streamlit/secrets.toml` in a location recognized by Streamlit. Keep
private values out of Git.

### Ollama says the model was not found

Run:

``` powershell
ollama list
```

If needed:

``` powershell
ollama pull qwen2.5:3b
```

Make sure the model name in `.env` or application settings matches the
installed model. Check whether `.env` overrides the code's default.

### Ollama connection errors

Check that Ollama is running and that the configured URL is
`http://localhost:11434`. Review the Ollama, Streamlit, and FastAPI
terminal logs for more detail.

### Qdrant collection not found

Check the collection name and confirm that indexing completed
successfully. Streamlit and FastAPI may use different collection names
or local database paths; a collection in one local database is not
automatically available in another.

### SQLite error: `cannot start a transaction within a transaction`

This can happen when multiple local Qdrant clients or concurrent writes
access the same local database. Stop the applications, verify their
paths, and use separate local directories or a shared Qdrant server.
Reuse the Streamlit client where appropriate and avoid simultaneous
indexing jobs.

### Weak or irrelevant answers

-   Confirm that the paper was uploaded and indexed successfully.
-   Ask a more specific question.
-   Check whether the PDF contains selectable text or scanned pages.
-   Review chunk size and overlap.
-   Verify the embedding model and collection configuration.
-   Inspect retrieved passages and application logs.
-   Verify important statements against the original paper.

### FastAPI returns `404 Not Found`

Open `http://127.0.0.1:8000/docs` and check the routes registered by the
current backend. A health endpoint is available only if it is
implemented.

## Limitations and Responsible Use

-   Generated answers and summaries can contain errors or omit important
    details.
-   Retrieval quality depends on text extraction, chunking, embeddings,
    and ranking.
-   Scanned PDFs may require OCR if the ingestion pipeline does not
    extract their text.
-   Local model performance depends on the model, hardware, memory, and
    prompt length.
-   Streamlit and FastAPI may use separate storage and ingestion
    workflows.
-   Generated output is not a substitute for checking the original
    research.
-   Respect copyright, data policies, and privacy requirements when
    uploading or sharing documents.

## Future Improvements

Potential improvements include:

-   Shared document indexing between Streamlit and FastAPI.
-   More robust PDF validation and OCR.
-   Source citations identifying the paper and page or passage
    supporting each answer.
-   Document metadata and filtering.
-   Automated evaluation of retrieval quality and answer faithfulness.
-   User authentication and access control for multi-user deployments.
-   Background document processing and progress indicators.
-   Logging, monitoring, tests, and production deployment configuration.

These are suggestions for future development, not claims that the
features are already implemented.


## License
MIT License 
