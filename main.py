import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import os

# File Paths
data_dir = 'flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
json_file = os.path.join(data_dir, 'cat_to_name.json')

# Load category mapping from json file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

# Define data transforms for train, validation, and test sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets with ImageFolder (assuming folder names represent class labels)
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Use a pre-trained model, ResNet50
model = models.resnet50(pretrained=True)

# Freeze parameters in pre-trained model to prevent retraining
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to match the number of classes in the dataset
num_classes = len(cat_to_name)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# Function to train the model
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels).item()

        val_loss /= len(valid_loader)
        accuracy /= len(valid_data)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5)

# Save the trained model
torch.save(model.state_dict(), 'flower_recognition_model.pth')

# Evaluate the model on the test dataset
def test_model(model, test_loader):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels).item()

    accuracy /= len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate the model
test_model(model, test_loader)
