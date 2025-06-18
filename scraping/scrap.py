from bing_image_downloader import downloader
import os
import shutil

# ✅ Waste categories with multiple search terms
waste_categories = {
    #"Plastic": "plastic waste",
    #"Glass": "glass waste",
    #"Metal": "metal scrap",
    #"Paper": "paper recycling waste",
        "E-Waste": [
        "electronic waste",
        "electronic trash",
        "old computer parts",
        "discarded electronics",
        "mobile phone e-waste",
        "scrap circuit boards"
    ]
}

# 📁 Output settings
output_dir = 'dataset'
images_per_keyword = 200  # images per keyword

# ✅ Download loop
for category, keyword_list in waste_categories.items():
    for keyword in keyword_list:
        print(f"🔍 Downloading: {keyword} → {category}")
        downloader.download(
            keyword,
            limit=images_per_keyword,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

        # 🧹 Move images into category folder
        original_folder = os.path.join(output_dir, keyword)
        target_folder = os.path.join(output_dir, category)

        if os.path.exists(original_folder):
            os.makedirs(target_folder, exist_ok=True)
            for file in os.listdir(original_folder):
                shutil.move(os.path.join(original_folder, file), target_folder)
            os.rmdir(original_folder)

print("✅ All images downloaded and sorted.")
