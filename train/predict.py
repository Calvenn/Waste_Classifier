import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import mimetypes

# 🔧 Settings
model_path = 'waste_classifier.pth'
category_names = ['Plastic', 'Glass', 'Metal', 'Paper', 'Cardboard']
img_size = 224
image_folder = 'test/trash'

# ✅ Image transformation
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(category_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Predict function
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return category_names[predicted.item()]

# ✅ Save correction
def save_correction(img_path, correct_label):
    save_dir = os.path.join('corrected_data', correct_label)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))
    print(f"✅ Saved corrected image to {save_dir}")

# ✅ Main execution
if __name__ == '__main__':
    all_files = sorted(os.listdir(image_folder))
    image_paths = []
    for file_name in all_files:
        file_path = os.path.join(image_folder, file_name)
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('image'):
            image_paths.append(file_path)
        if len(image_paths) == 10:
            break

    for img_path in image_paths:
        print(f"\n🖼️ Predicting: {img_path}")
        prediction = predict_image(img_path)
        if prediction is None:
            continue

        print(f"🧠 Predicted category: {prediction}")
        correct = input("Is this correct? (y/n): ").strip().lower()

        if correct != 'y':
            new_label = input(f"Enter the correct category from {category_names}: ").strip()
            if new_label in category_names:
                save_correction(img_path, new_label)
            else:
                print("❌ Invalid category. Not saved.")
        else:
            print("👍 No correction needed.")

        # 🗑️ Delete after response
        try:
            os.remove(img_path)
            print(f"🗑️ Deleted: {img_path}")
        except Exception as e:
            print(f"❌ Failed to delete {img_path}: {e}")
