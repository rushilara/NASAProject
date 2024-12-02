import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

# Configuration
new_data_dir = "../images/attempt2_augmented_training_set"  # Path to the new dataset directory
model_save_path = "./models/v3_efficientnet_b2_finetuned_new.pth"
pretrained_model_path = "./models/v2_efficientnet_b2_finetuned.pth"
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-5  # L2 regularization
dropout_rate = 0.3
num_classes = 8  # Change this if the new dataset has a different number of classes
epochs = 25
patience = 5  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=new_data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

# Load EfficientNet-B2 with pretrained weights
model = EfficientNet.from_name('efficientnet-b2')
num_features = model._fc.in_features

# Modify the model for fine-tuning
model._fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(num_features, num_classes)
)

# Load pretrained weights
model.load_state_dict(torch.load(pretrained_model_path))
print(f"Pretrained weights loaded from {pretrained_model_path}")

model = model.to(device)

# Freeze BatchNorm layers to stabilize training
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

# Early stopping variables
best_val_loss = float('inf')
early_stopping_counter = 0

# Training and validation loop
for epoch in range(epochs):
    print(f"Epoch [{epoch + 1}/{epochs}]")
    
    # Training
    model.train()
    train_loss, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = correct / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_dataset)
    
    # Print results for this epoch
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Learning rate scheduling
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model improved. Saved to {model_save_path}")
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{patience}")
        if early_stopping_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

print("Training complete.")
