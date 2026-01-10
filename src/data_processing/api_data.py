import aiohttp
import asyncio
import os
import logging
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.read_configs import read_config

# Config
config = read_config('configs/base_config.yml')
api_cfg = config['api']

BASE_API_URL = api_cfg["base_api_url"]
PUBLISHERS = api_cfg["publisher"]
OUTPUT_DIR = api_cfg["output_dir"]
CONCURRENCY_LIMIT = api_cfg["concurrency_limit"]
START_YEAR = api_cfg["start_year"]
END_YEAR = api_cfg["end_year"]

TARGET_TYPES = {
    "Ustawa", 
    "Rozporządzenie", 
    "Obwieszczenie", 
    "Uchwała", 
    "Umowa międzynarodowa"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def download_pdf(session, act, semaphore):
    """Pobiera plik PDF na podstawie parametrów aktu."""
    act_type = act.get("type")
    if act_type not in TARGET_TYPES:
        return

    year = act.get("year")
    pos = act.get("pos")
    vol = act.get("volume")
    pub = act.get("publisher")
    
    if vol:
        # < 2012 with volume
        pdf_url = f"{BASE_API_URL}/acts/{pub}/{year}/volumes/{vol}/{pos}/text.pdf"
    else:
        # > 2012 without volume
        pdf_url = f"{BASE_API_URL}/acts/{pub}/{year}/{pos}/text.pdf"
    
    save_dir = os.path.join(OUTPUT_DIR, str(year), act_type)
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f"{pub}_{year}_{vol or 0}_{pos}.pdf"
    file_path = os.path.join(save_dir, file_name)

    if os.path.exists(file_path):
        return

    pdf_headers = {"Accept": "application/pdf"}

    async with semaphore:
        try:
            async with session.get(pdf_url, headers=pdf_headers) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    if content.startswith(b'%PDF'):
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        logging.info(f"Pobrano: {act_type} {pub}/{year}/poz.{pos}")
                    else:
                        error_msg = content.decode('utf-8', errors='ignore')[:200]
                        logging.warning(f"Serwer zwrócił tekst zamiast PDF dla {pos}: {error_msg}")
                else:
                    logging.warning(f"Brak PDF (Status {response.status}): {pdf_url}")
        except Exception as e:
            logging.error(f"Błąd sieciowy przy {pdf_url}: {e}")

async def fetch_year_items(session, pub, year):
    """Pobiera listę aktów dla danego roku."""
    url = f"{BASE_API_URL}/acts/{pub}/{year}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                items = data.get("items", [])
                logging.info(f"Wydawca {pub}, Rok {year}: Znaleziono {len(items)} aktów.")
                return items
            return []
    except Exception as e:
        logging.error(f"Błąd pobierania listy {pub}/{year}: {e}")
        return []

async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Expert-Crawler"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        start_year = api_cfg.get("start_year", 2020)
        end_year = api_cfg.get("end_year", 2024)
        
        for year in range(start_year, end_year + 1):
            for pub in PUBLISHERS:
                acts = await fetch_year_items(session, pub, year)
                if acts:
                    tasks = [download_pdf(session, act, semaphore) for act in acts]
                    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())