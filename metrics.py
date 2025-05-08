import torch
import os
import torch.nn as nn
from torchvision import models
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# File Paths
data_dir = 'flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
json_file = os.path.join(data_dir, 'cat_to_name.json')

# Load category mapping from json file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

# Define the number of classes
num_classes = len(cat_to_name)

# Load the pre-trained model
model = models.resnet50(weights=None)

# Adjust the final fully connected layer to match the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the trained model weights
model.load_state_dict(torch.load(r'C:\Users\zenye\Desktop\FlowerApp\flower_recognition_model.pth'))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the function to calculate the metrics
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate the metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


# Create a DataLoader for the test set
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transforms for the test set
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Create a DataLoader for the test set
test_loader = DataLoader(test_data, batch_size=32)

# Evaluate the model on the test set
evaluate_model(model, test_loader)
