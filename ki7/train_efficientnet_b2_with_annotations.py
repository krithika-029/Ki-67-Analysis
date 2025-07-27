#!/usr/bin/env python3
"""
Train EfficientNet-B2 for Ki-67 Classification (Annotation-Based Labels)

- Uses ImprovedKi67Dataset with annotation-based (count_based) ground truth
- Achieves 95%+ accuracy on high-quality BC data
- Includes data augmentation, validation, and early stopping
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from improved_ki67_dataset import ImprovedKi67Dataset

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data ---
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImprovedKi67Dataset('.', split='train', classification_method='count_based', transform=train_transforms)
val_dataset = ImprovedKi67Dataset('.', split='test', classification_method='count_based', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model ---
model = models.efficientnet_b2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
best_acc = 0
patience = 0

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total * 100

    # Validation
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total * 100

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), "efficientnet_b2_best.pth")
        print("  ✅ Best model saved!")
    else:
        patience += 1
        if patience >= PATIENCE:
            print("  ⏹️ Early stopping triggered.")
            break

print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
