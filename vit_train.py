import os
import random
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, AdamW
from sklearn.model_selection import KFold
from PIL import Image

# Paths and parameters
data_folder = "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/src/frames_output"
train_folder = "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/src/frames_output/train_data"
test_folder = "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/src/frames_output/test_data"
k = 10  # Number of folds for cross-validation
split_ratio = 0.7  # Train-test split ratio

# Step 1: Shuffle and Split Images into Train and Test Folders with Label List
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

label_list = []  # List to store (image_path, label) tuples

for label_folder in os.listdir(data_folder):
    # Skip "test_data" and "train_data" folders
    if label_folder in ["test_data", "train_data"]:
        continue

    label_path = os.path.join(data_folder, label_folder)
    if os.path.isdir(label_path):
        label = int(label_folder.split("_")[1])  # Extract label from folder name (e.g., "label_0" -> 0)
        images = os.listdir(label_path)
        random.shuffle(images)
        
        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]
        
        # Create labeled subdirectories in train and test folders
        train_label_folder = os.path.join(train_folder, label_folder)
        test_label_folder = os.path.join(test_folder, label_folder)
        os.makedirs(train_label_folder, exist_ok=True)
        os.makedirs(test_label_folder, exist_ok=True)
        
        # Copy images and add to label list
        for image in train_images:
            src = os.path.join(label_path, image)
            dst = os.path.join(train_label_folder, image)
            shutil.copy(src, dst)
            label_list.append((dst, label))  # Add (image_path, label) to label list
        
        for image in test_images:
            src = os.path.join(label_path, image)
            dst = os.path.join(test_label_folder, image)
            shutil.copy(src, dst)

# Step 2: Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, label_list, transform=None):
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path, label = self.label_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Step 3: Prepare 10-Fold Cross-Validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 4: Training Loop for Each Fold
num_epochs = 5
batch_size = 32

for fold_idx, (train_indices, val_indices) in enumerate(kf.split(label_list)):
    print(f"\nTraining fold {fold_idx + 1}/{k}")

    # Create subsets for the fold
    train_subset = [label_list[i] for i in train_indices]
    val_subset = [label_list[i] for i in val_indices]
    
    train_dataset = CustomImageDataset(train_subset, transform=transform)
    val_dataset = CustomImageDataset(val_subset, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=len(set([label for _, label in label_list]))ignore_mismatched_sizes=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training and Validation for the fold
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / len(val_dataset)
        print(f"Fold {fold_idx+1}, Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {total_train_loss/len(train_loader):.4f}, "
              f"Val Loss: {total_val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
