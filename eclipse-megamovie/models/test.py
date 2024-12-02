import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from PIL import Image

# Configuration
model_path = "./models/v4_efficientnet_b2_finetuned_new.pth"
test_data_dir = "../images/test"  # Path to test images
output_csv_path = "./v7predictions.csv"
batch_size = 16
num_classes = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations for test dataset
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset for test folder without subdirectories
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)
                            if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)  # Return image and its filename

# Test dataset and DataLoader
test_dataset = TestDataset(folder_path=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = EfficientNet.from_name('efficientnet-b2')
num_features = model._fc.in_features
model._fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, num_classes)
)  # Match the fine-tuned model's _fc structure
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluate and save predictions
def evaluate_model():
    predictions = []
    with torch.no_grad():
        for inputs, filenames in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Save predictions with image filenames
            for filename, pred in zip(filenames, preds):
                predictions.append((filename, pred.item()))
    
    # Write predictions to CSV
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "label"])
        writer.writerows(predictions)
    print(f"Predictions saved to {output_csv_path}")

# Run evaluation
evaluate_model()
