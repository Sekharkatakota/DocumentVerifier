import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai

from backend.custom_types import RAGChunkAndSRC, RAGQueryResult, RAGSearchResult, RAGUPsertResult
from backend.data_loader import embed_texts, load_and_chunk_pdf
from backend.vector_db import QdrantStorage


load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load() -> RAGChunkAndSRC:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSRC(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSRC) -> RAGUPsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id or "unknown"
        vecs = embed_texts(chunks)
        ids = [f"{source_id}:{i}" for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUPsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunks-pdf",
        _load,
        output_type=RAGChunkAndSRC,
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUPsertResult,
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data.get("question")
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {context}" for context in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    result = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = result["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# Initialize FastAPI
app = FastAPI(
    title="Document Verifier API",
    description="RAG-based PDF document verification and question answering",
    version="1.0.0"
)


# Root endpoint
@app.get("/")
async def home():
    return {
        "status": "success",
        "message": "Document Verifier API is running!",
        "framework": "FastAPI",
        "services": ["RAG", "Qdrant", "OpenAI", "Inngest"]
    }


# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "Backend is running successfully"
    }


# Register Inngest functions
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai]
)


# Local development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)