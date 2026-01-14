from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid
from src.data_processing.ocr import OCR # Twoja klasa

app = FastAPI(title="ISAP OCR Service")

# Inicjalizacja Twojej klasy OCR
ocr_processor = OCR('configs/base_config.yml')

# Folder tymczasowy na przychodzące pliki
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ocr")
async def process_pdf(file: UploadFile = File(...)):
    # Sprawdzenie rozszerzenia
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Tylko pliki PDF są obsługiwane.")

    # Zapis pliku tymczasowo na dysk (fitz potrzebuje ścieżki lub bytes)
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Wywołanie Twojej metody read_pdf
        text = ocr_processor.read_pdf(temp_path)
        
        return {
            "filename": file.filename,
            "text_length": len(text),
            "content": text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Sprzątanie - usuwamy plik tymczasowy
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)