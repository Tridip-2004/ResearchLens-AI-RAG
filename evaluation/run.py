import uvicorn

from app.config import settings


def main():

    print("-" * 50)

    print("ResearchLens")
    print("Research Document Question Answering System")

    print("-" * 50)

    print(f"Application: {settings.app_name}")
    print(f"Version: {settings.app_version}")

    print(f"LLM: {settings.ollama_model}")

    print(f"Embedding Model: {settings.embedding_model}")

    print(f"Database: {settings.collection_name}")

    print("-" * 50)

    print("Starting FastAPI server...")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )


if __name__ == "__main__":
    main()