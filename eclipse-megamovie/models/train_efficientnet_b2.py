import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os

# Paths
DATASET_PATH = "../images/augmented_2000"
MODEL_SAVE_PATH = "./models/efficientnet_b2_augmented_2000.pth"
CHECKPOINT_DIR = "./models/checkpoints"
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_CLASSES = 8  # Number of classes

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure directories exist
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

# Ensure dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found. Please check the path.")

# Data transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Loaded dataset with {len(dataset)} images.")

# Initialize EfficientNet-B2 model
model = EfficientNet.from_name('efficientnet-b2')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)  # Adjust output layer
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop with checkpointing
def train_model(model, dataloader, criterion, optimizer, epochs, save_path, checkpoint_dir):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")  # Print the epoch number first
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Epoch stats
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"efficientnet_b2_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Final model save
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Train and save the model
if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, EPOCHS, MODEL_SAVE_PATH, CHECKPOINT_DIR)
