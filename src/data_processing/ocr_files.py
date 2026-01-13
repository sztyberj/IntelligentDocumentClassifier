import os
import json
import sys
from tqdm import tqdm
import logging

# Dodanie ścieżki projektu do sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing.ocr import OCR

# Konfiguracja logowania błędów
logging.basicConfig(
    filename='ocr_errors.log', 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Przetwarza pliki PDF przez OCR i zapisuje wynik do JSONL z uproszczonymi metadanymi.
    """
    docs_root_dir = './isap_docs'
    output_file = 'raw_data.jsonl'
    config_path = 'configs/base_config.yml'

    try:
        ocr = OCR(config_path=config_path)
    except Exception as e:
        print(f"Błąd inicjalizacji OCR: {e}")
        return

    # Otwarcie pliku w trybie dopisywania ('a')
    with open(output_file, 'a', encoding='utf-8') as f:
        # Iteracja po latach
        for year_dir in tqdm(os.listdir(docs_root_dir), desc="Przetwarzanie lat"):
            year_path = os.path.join(docs_root_dir, year_dir)
            if not os.path.isdir(year_path):
                continue

            # Iteracja po typach dokumentów
            for type_dir in os.listdir(year_path):
                type_path = os.path.join(year_path, type_dir)
                if not os.path.isdir(type_path):
                    continue

                # Iteracja po plikach PDF
                for filename in os.listdir(type_path):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(type_path, filename)
                        
                        try:
                            # Próba odczytu tekstu (OCR)
                            text = ocr.read_pdf(file_path)

                            if text and text.strip():
                                # Nowy schemat metadanych bez 'pos'
                                data = {
                                    'text': text.strip(),
                                    'meta': {
                                        'type': type_dir,
                                        'year': int(year_dir) if year_dir.isdigit() else year_dir,
                                        'source_file': filename,
                                        'path': file_path
                                    }
                                }
                                # Zapis do JSONL
                                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                                
                        except AttributeError as e:
                            if 'witdh' in str(e):
                                logging.error(f"Błąd 'witdh' w bibliotece przy pliku {file_path}. Pomijam.")
                            else:
                                logging.error(f"AttributeError przy {file_path}: {e}")
                        except Exception as e:
                            logging.error(f"Błąd krytyczny przy {file_path}: {e}")

if __name__ == '__main__':
    main()