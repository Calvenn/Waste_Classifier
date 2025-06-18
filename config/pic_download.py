import pandas as pd
import os
import requests
from tqdm import tqdm

# Path to your CSV file
csv_path = 'dataset/labeled_dataset.csv'

# Load CSV
df = pd.read_csv(csv_path)

# Ensure 'image_url' and 'category' columns exist
if 'image_url' not in df.columns or 'category' not in df.columns:
    raise ValueError("CSV must contain 'image_url' and 'category' columns")

# Loop through each row and download
for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row['image_url']
    category = row['category']
    save_dir = os.path.join('test', category)
    os.makedirs(save_dir, exist_ok=True)

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # Save image with index as filename
            ext = os.path.splitext(url)[1].split('?')[0] or '.jpg'
            filename = f"{idx}{ext}"
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"❌ Failed to download {url} (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
