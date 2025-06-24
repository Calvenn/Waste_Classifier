import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
from sklearn.metrics import classification_report

# Settings (loads img from dataset folder)
data_dir = 'dataset'
batch_size = 32
num_epochs = 15
img_size = 224 # Resize to 224x224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU when available

# Transforms
transform_train = transforms.Compose([ # With data augmentation to improve generalization (Train data)
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
]) 
transform_val_test = transforms.Compose([ # Only resize and normalizes images (Test data)
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset and class info
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_val_test)
class_names = full_dataset.classes

# Class Weights for Imbalanced Data
label_counts = Counter([label for _, label in full_dataset])
max_count = max(label_counts.values())
class_weights = [max_count / label_counts[i] for i in range(len(class_names))]
weights_tensor = torch.FloatTensor(class_weights).to(device)

# Dataset split
total_size = len(full_dataset)
train_size = int(0.7 * total_size) # 70% training
val_size = int(0.15 * total_size) # 15% validation
test_size = total_size - train_size - val_size # 15% testing
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_dataset.dataset.transform = transform_train # Apply data augmentation

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model
model = models.resnet18(pretrained=True) # Loads CNN
model.fc = nn.Linear(model.fc.in_features, len(class_names)) 
model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
print(f"Classes: {class_names}")

# Training loop
for epoch in range(num_epochs):
    model.train() # Start training
    running_loss = 0.0
    correct = 0
    total = 0
    total_conf = 0.0

    for images, labels in train_loader:
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
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total_conf += probs[range(len(predicted)), predicted].sum().item()

    scheduler.step()

    train_acc = 100 * correct / total
    avg_conf = (total_conf / total) * 100
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Avg Conf: {avg_conf:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# Final Testing
model.eval()
test_correct = 0
test_total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

test_acc = 100 * test_correct / test_total
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save model
torch.save(model.state_dict(), 'waste_classifier.pth')
print("Model saved as 'waste_classifier.pth'")
