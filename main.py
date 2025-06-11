import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import fitz
import io

from handlers import RAGHandler, FileHandler

app = FastAPI()

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


@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.post("/v1/rag/upload")
async def upload_rag(file: UploadFile = File(...)):
    url = None
    title = file.filename
    # TODO: generate therapeutic area, might call OpenAI with some part of the file
    therapeutic_area = None

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
    return {"text": text}
