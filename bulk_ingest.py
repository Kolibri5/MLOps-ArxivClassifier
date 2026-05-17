import arxiv
import json
import os
from datetime import datetime

def fetch_bulk_data():
    categories = ['cs.AI', 'cs.CV', 'cs.LG']
    all_papers = []
    
    print("Mulai menarik 1.500 data historis (Kaggle-like sample)... Ini mungkin memakan waktu 1-2 menit.")
    
    for cat in categories:
        print(f"Menarik 500 paper untuk kategori {cat}...")
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=500,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        client = arxiv.Client()
        for result in client.results(search):
            paper = {
                'paper_id': result.entry_id,
                'title': result.title,
                'abstract': result.summary,
                'primary_category': cat,
                'published_date': result.published.strftime('%Y-%m-%d')
            }
            all_papers.append(paper)

    # Simpan ke folder raw
    os.makedirs('data/raw', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/arxiv_bulk_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=4)
        
    print(f"Selesai! 1.500 paper berhasil disimpan di {filename}")

if __name__ == "__main__":
    fetch_bulk_data()