import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
import csv

# Configuration
test_data_dir = "../images/test"  # Path to the test dataset
model_path = "./models/940_efficientnet_b1_20241126-212325.pth"  # Path to the trained model
num_classes = 8  # Number of bins/classes
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid image extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# Data transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize for EfficientNet-B1
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Function to load images from a flat directory structure
def load_images_from_directory(directory, transform, valid_extensions):
    images = []
    image_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath).convert("RGB")  # Open image and convert to RGB
                images.append(transform(img))
                image_paths.append(filepath)
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
    return torch.stack(images), image_paths

# Load images and their file paths
test_images, test_image_paths = load_images_from_directory(test_data_dir, transform, valid_extensions)

# Create DataLoader for test images
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size)

# Initialize the EfficientNet-B1 model
model = EfficientNet.from_name('efficientnet-b1')  # Create the EfficientNet-B1 architecture
model._fc = nn.Linear(model._fc.in_features, num_classes)  # Update the final layer for 8 classes
model.load_state_dict(torch.load(model_path))  # Load the trained model weights
model = model.to(device)
model.eval()

# Perform inference
results = []
with torch.no_grad():
    for batch_idx, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions = outputs.argmax(1).cpu().numpy()

        # Map predictions to image file paths
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(predictions)
        for i, prediction in enumerate(predictions):
            image_path = os.path.basename(test_image_paths[start_idx + i])  # Get just the file name
            results.append((image_path, prediction))

# Save predictions to CSV
output_csv = "./models/test_predictions.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "label"])  # Ensure 'label' column is used
    writer.writerows(results)

print(f"Predictions saved to {output_csv}")
