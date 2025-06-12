import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from handlers import ChromaRAGHandler, FileHandler, GPTHandler

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


# need to add error handleing
@app.post("/v1/rag/upload")
async def upload_rag(file: UploadFile = File(...)):
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

    # TODO: need to initialize RAGHandler with user_id
    ChromaRAGHandler().save_to_vector_database(text, metadata)
    return {"text": therapeutic_area}
