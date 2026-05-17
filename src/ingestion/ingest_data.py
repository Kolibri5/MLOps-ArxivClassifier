import arxiv
import json
import os
from datetime import datetime

def run_ingestion():
    print("Memulai proses Ingestion data dari API arXiv...")
    
    search_query = 'cat:cs.AI OR cat:cs.CV OR cat:cs.LG'
    
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3, # Memberikan jeda 3 detik untuk menghindari blokir
        num_retries=5    # Jika ditolak (error 429), otomatis coba lagi hingga 5 kali
    )
    
    search = arxiv.Search(
        query=search_query,
        max_results=50,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    data_raw = []
    
    try:
        # Client sekarang akan menangani retry dan delay secara otomatis
        for result in client.results(search):
            data_raw.append({
                "paper_id": result.get_short_id(),
                "title": result.title,
                "abstract": result.summary,
                "published_date": result.published.strftime("%Y-%m-%d %H:%M:%S"),
                "primary_category": result.primary_category,
                "authors": [author.name for author in result.authors]
            })
    except Exception as e:
        print(f"\nGagal mengambil data dari server. Error detail: {e}")
        print("Saran: Tunggu 1-2 menit, lalu jalankan ulang skrip.")
        return
        
    if not data_raw:
        print("Tidak ada data yang berhasil ditarik.")
        return

    # Menyimpan data dengan timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/arxiv_raw_{current_time}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_raw, f, indent=4, ensure_ascii=False)
        
    print(f"Ingestion Selesai! {len(data_raw)} paper berhasil disimpan di {file_path}")

if __name__ == "__main__":
    run_ingestion()