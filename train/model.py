import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ✅ Settings
data_dir = 'dataset'  # your folder containing subfolders of categories
batch_size = 32
num_epochs = 10
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Data transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ Define model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

# ✅ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
print(f"Training on {len(dataset)} images in {len(dataset.classes)} categories: {dataset.classes}")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 🔍 Confidence per batch (average probability of predicted class)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences = probs[range(len(predicted)), predicted]
        avg_confidence = confidences.mean().item() * 100
        print(f"  Batch {batch_idx+1}/{len(train_loader)} | Avg Confidence: {avg_confidence:.2f}%")

    acc = 100 * correct / total
    print(f"✅ Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

# ✅ Save model
torch.save(model.state_dict(), 'waste_classifier.pth')
print("✅ Model saved as 'waste_classifier.pth'")
