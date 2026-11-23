from pathlib import Path
from collections import Counter
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import parse_file, chunk_documents
from app.store import get_vector_store
from app.retrieval import get_retriever
from app.answer import generate_answer


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "ResearchLens document question-answering API"
    )
)


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


@app.get("/")
def home():

    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "message": "ResearchLens API is ready"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "model": settings.ollama_model
    }


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md"
    }

    original_name = Path(
        file.filename or ""
    ).name

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported document format."
        )

    if not original_name:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    upload_directory = Path(
        settings.upload_dir
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use a unique filename to avoid overwriting uploads.
    stored_name = (
        f"{uuid.uuid4().hex[:12]}_{original_name}"
    )

    file_path = upload_directory / stored_name

    max_bytes = settings.max_upload_mb * 1024 * 1024

    total_size = 0

    try:

        with open(file_path, "wb") as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > max_bytes:

                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file is too large."
                    )

                output.write(chunk)

        documents = parse_file(
            file_path
        )

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No readable text found."
            )

        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="Document contains no usable text."
            )

        store = get_vector_store()

        count = store.add_documents(
            chunks
        )

        return {
            "status": "success",
            "filename": stored_name,
            "original_filename": original_name,
            "pages": len(documents),
            "chunks": count,
            "message": "Document uploaded and indexed."
        }

    except HTTPException:

        file_path.unlink(
            missing_ok=True
        )

        raise

    except Exception as error:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        await file.close()


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    try:

        retriever = get_retriever()

        documents = retriever.retrieve(
            query=request.question,
            top_k=request.top_k
        )

        result = generate_answer(
            question=request.question,
            documents=documents
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "mode": result["mode"],
            "retrieved_chunks": len(documents)
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/documents")
def list_documents():

    try:

        store = get_vector_store()

        documents = store.get_all_documents()

        filenames = [
            document["metadata"].get(
                "source",
                "Unknown"
            )
            for document in documents
        ]

        counts = Counter(filenames)

        return {
            "total_documents": len(counts),
            "total_chunks": len(documents),
            "documents": [
                {
                    "filename": filename,
                    "chunks": count
                }
                for filename, count in counts.items()
            ]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.delete("/documents/{filename}")
def delete_document(filename: str):

    filename = Path(filename).name

    try:

        store = get_vector_store()

        store.delete_document(
            filename
        )

        file_path = (
            Path(settings.upload_dir) / filename
        )

        if file_path.exists():
            file_path.unlink()

        return {
            "status": "success",
            "message": "Document deleted successfully."
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )