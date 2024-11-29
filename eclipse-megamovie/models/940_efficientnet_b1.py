import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split
import datetime

# Configuration
data_dir = "../images/enhanced_grouped_images"  # Path to the dataset
num_classes = 8  # Number of bins/classes
batch_size = 16
learning_rate = 1e-4  # Adjusted for smaller dataset
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create models directory if it doesn't exist
if not os.path.exists("./models"):
    os.makedirs("./models")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize for EfficientNet-B1
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

# Print dataset sizes
print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

# Initialize the EfficientNet-B1 model
try:
    model = EfficientNet.from_pretrained('efficientnet-b1')  # Use pre-trained weights
    print("EfficientNet-B1 pre-trained weights loaded successfully.")
except Exception as e:
    print(f"Error loading pre-trained weights: {e}")
    exit(1)

model._fc = nn.Linear(model._fc.in_features, num_classes)  # Update the final layer for 8 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
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

    val_loss, val_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {correct / len(train_dataset):.4f}")
    print(f"Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_correct / len(val_dataset):.4f}")

# Save the trained model with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"./models/940_efficientnet_b1_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model training complete. Model saved as {model_path}")
