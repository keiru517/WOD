from fastapi import UploadFile, HTTPException
import fitz
import io
from docx import Document


class FileHandler:
    def __init__(self):
        pass

    async def process_pdf(self, file: UploadFile):
        content = await file.read()
        pdf_stream = io.BytesIO(content)

        try:
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

        return text

    async def process_txt(self, file: UploadFile):
        try:
            content = await file.read()
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin1")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to decode TXT: {e}"
                )
        return text

    async def process_docx(self, file: UploadFile):
        content = await file.read()
        docx_stream = io.BytesIO(content)

        try:
            document = Document(docx_stream)
            text = "\n".join([para.text for para in document.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {e}")

        return text
