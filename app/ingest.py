from pathlib import Path

import pymupdf
from docx import Document


def read_pdf(file_path):

    documents = []

    with pymupdf.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_number
                }
            })

    return documents


def read_docx(file_path):

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    for table in document.tables:

        for row in table.rows:

            values = [
                cell.text.strip()
                for cell in row.cells
            ]

            paragraphs.append(" | ".join(values))

    full_text = "\n".join(paragraphs)

    if not full_text.strip():
        return []

    return [{
        "text": full_text,
        "metadata": {
            "source": Path(file_path).name,
            "page": None
        }
    }]


def read_text(file_path):

    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="replace"
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "page": None
        }
    }]


def parse_file(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    if extension == ".docx":
        return read_docx(file_path)

    if extension in [".txt", ".md"]:
        return read_text(file_path)

    raise ValueError(
        f"Unsupported file format: {extension}"
    )


def split_text(text, chunk_size=1000, overlap=200):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk overlap.")

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            boundary = text.rfind(
                " ",
                start + chunk_size // 2,
                end
            )

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def chunk_documents(
    documents,
    chunk_size=1000,
    overlap=200
):

    chunked_documents = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        chunks = split_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(chunks):

            chunked_documents.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk": index
                }
            })

    return chunked_documents