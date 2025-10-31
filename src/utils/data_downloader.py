import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
output_dir = "data/raw/sleep-cassette"
os.makedirs(output_dir, exist_ok=True)

# Get all EDF links
response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

file_links = [
    link.get("href")
    for link in soup.find_all("a")
    if link.get("href") and link.get("href").endswith(".edf")
]

output_file = "links.txt"
# Write all links to text file
with open(output_file, "w") as f:
    for url in file_links:
        f.write(url + "\n")


def download_file(href):
    file_url = base_url + href
    out_path = os.path.join(output_dir, href)
    if os.path.exists(out_path):
        return href, "exists"

    try:
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return href, "ok"
    except Exception as e:
        return href, f"error: {e}"


# Limit number of threads to avoid overloading server
max_workers = 4

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_file, href) for href in file_links]
    for i, future in enumerate(
        tqdm(as_completed(futures), total=len(futures), desc="Downloading all")
    ):
        href, status = future.result()
        tqdm.write(f"[{i+1}/{len(file_links)}] {href}: {status}")

print("\nAll downloads completed")
