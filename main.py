import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from handlers import ChromaRAGHandler, FileHandler, GPTHandler

app = FastAPI(
    title="WOD RAG API",
    description="API for the WOD RAG",
    version="1.0.0",
    terms_of_service="https://www.wod.com/terms",
    contact={
        "name": "WOD",
        "url": "https://www.wod.com",
        "email": "contact@wod.com",
    },
)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/v1/health",
    summary="Health Check",
    description="Check if the API is running",
    operation_id="health",
)
async def health():
    return {"message": "API is running"}


@app.post(
    "/api/v1/rag/upload",
    summary="Upload File",
    description="Upload a file to the RAG (pdf, txt, docx)",
    operation_id="upload_rag",
)
async def upload_rag(user_id: str = "global", file: UploadFile = File(...)):
    url = None
    title = file.filename
    timestamp = datetime.now().strftime("%m-%Y")

    if file.content_type == "application/pdf":
        text = await FileHandler().process_pdf(file)
    elif file.content_type == "text/plain":
        text = await FileHandler().process_txt(file)
    elif (
        file.content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        text = await FileHandler().process_docx(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    THERAPEUTIC_PROMPT = os.getenv("THERAPEUTIC_PROMPT")
    therapeutic_area = GPTHandler(openai_api_key).generate_response(
        THERAPEUTIC_PROMPT, text[:3000]  # 500 words
    )

    # store data in vector database with RAGHandler
    metadata = {
        "url": url,
        "title": title,
        "therapeutic_area": therapeutic_area,
        "timestamp": timestamp,
    }

    status, result = ChromaRAGHandler().save_to_vector_database(text, metadata, user_id)
    return {"status": status, "result": result}


@app.post(
    "/api/v1/rag/delete/file",
    summary="Delete File",
    description="Delete a file from the RAG",
    operation_id="delete_file",
)
async def delete_file(file_name: str, user_id: str = "global"):
    status, result = ChromaRAGHandler().delete_file_by_name_from_vector_database(
        file_name, user_id
    )
    return {"status": status, "result": result}


@app.post(
    "/api/v1/rag/delete/collection",
    summary="Delete Collection",
    description="Delete a collection from the RAG",
    operation_id="delete_collection",
)
async def delete_collection(user_id: str = "global"):
    status, result = ChromaRAGHandler().delete_collection_from_vector_database(user_id)
    return {"status": status, "result": result}


@app.post(
    "/api/v1/rag/query",
    summary="Query the RAG",
    description="Query the RAG with a question",
    operation_id="query",
)
async def query_rag(metadata: dict, query: str, user_id: str = "global"):
    status, result = ChromaRAGHandler().query_vector_database(metadata, query, user_id)
    return {"status": status, "result": result}
