import os
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import models, transforms

# Load model
category_names = ['Plastic', 'Glass', 'Metal', 'Paper', 'E-Waste']
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(category_names))
model.load_state_dict(torch.load("waste_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Flask app
app = Flask(__name__)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    image = Image.open(request.files['file']).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(probs, 1)
        predicted_label = category_names[predicted.item()]
        confidence = probs[0][predicted.item()].item()

    return jsonify({
        'prediction': predicted_label,
        'confidence': f"{confidence*100:.2f}%"
    })

@app.route('/')
def home():
    return "AI Waste Classifier API is running."

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000) run locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
