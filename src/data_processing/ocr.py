import fitz
import pytesseract
from PIL import Image
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.read_configs import read_config

class OCR:
    def __init__(self, config_path):
        config = read_config(config_path)
        config_ocr = config['ocr']
        page_limit = config_ocr['page_limit']
        min_words = config_ocr['min_words']

        self.PAGE_LIMIT = min_words
        self.MIN_WORDS = page_limit

    def read_pdf(self, path):
            """
            Reading pdf docs
            
            :param self: 
            :param path: Doc path
            """

            try:
                doc = fitz.open(path)
                page_count = len(doc)

                if page_count > self.PAGE_LIMIT:
                    indices = [0, 1, 2, page_count-3, page_count-2, page_count-1]
                else:
                    indices = list(range(page_count))
                
                text_parts = []

                for i in indices:
                    page = doc.load_page(i)
                    text_parts.append(page.get_text().strip())

                full_text = " ".join(text_parts)

                if len(full_text) > self.MIN_WORDS:
                    doc.close()
                    return full_text
                
                print(f"-> START OCR {os.path.basename(path)}")
                ocr_parts = []

                for i in indices:
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                    img = Image.frombytes("RGB", [pix.witdh, pix.height], pix.samples)
                
                    ocr_parts.append(pytesseract.image_to_string(img, lang="pol"))

                doc.close()
                return " ".join(ocr_parts)
            
            except Exception as e:
                print(f"-> Reading ERROR {path}: {e}")
                return ""

if __name__ == "__main__":
    ocr = OCR('configs/base_config.yml')
    while True:
        print("Podaj ścieżkę do pliku: ")
        file = input(str)
        output = ocr.read_pdf(file)
        print(output)
        print(" ")
        print("==========================================")