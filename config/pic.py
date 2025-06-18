import pandas as pd
import os
import requests
from tqdm import tqdm

# Load your annotated CSV
df = pd.read_csv('dataset/labeled_dataset_v1.csv')  # replace with your file name
output_dir = 'dataset'

# Create folders for each category (from 'label' column)
for label in df['label'].unique():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Download images into their labeled folders
for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row['image_url']
    label = row['label']
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            ext = url.split('.')[-1].split('?')[0]
            filename = os.path.join(output_dir, label, f'{idx}.{ext}')
            with open(filename, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
