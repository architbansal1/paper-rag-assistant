"""
Download curated arXiv papers as PDFs into data/papers/.

Run this first, before ingest.py.
"""
import os
import time
import urllib.request

PAPERS = {
    "attention": "1706.03762",
    "lora": "2106.09685",
    "rag": "2005.11401",
    "protonet": "1703.05175",
    "matching_net": "1606.04080",
    "gptq": "2210.17323",
    "smoothquant": "2211.10438",
    "distillation": "1503.02531",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")


def download_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, arxiv_id in PAPERS.items():
        out_path = os.path.join(OUT_DIR, f"{name}.pdf")
        if os.path.exists(out_path):
            print(f"[skip] {name} already downloaded")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"[download] {name} <- {url}")
        try:
            urllib.request.urlretrieve(url, out_path)
            time.sleep(1)  # be polite to arXiv
        except Exception as e:
            print(f"  FAILED: {e}")
    print("Done.")


if __name__ == "__main__":
    download_all()
