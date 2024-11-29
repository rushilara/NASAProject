import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("best_model.pth", map_location=device)
model.eval()

# Define test data transformations
test_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define label-to-number mapping
label_mapping = {
    0: "TotalSolarEclipse",
    1: "0to25percentPartialEclipse",
    2: "26to55percentPartialEclipse",
    3: "56to95percentPartialEclipse",
    4: "Darks",
    5: "DiamondRing_BaileysBeads_SolarEclipse",
    6: "Flats",
    7: "NotASolarEclipse",
}
number_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping

# Prepare test data
test_dir = "test"
test_images = sorted(os.listdir(test_dir))  # Assuming test files are in the "test/" folder
test_paths = [os.path.join(test_dir, img) for img in test_images if img.lower().endswith(('png', 'jpg', 'jpeg'))]

# Generate predictions
predictions = []
for img_path in test_paths:
    img = Image.open(img_path).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predictions.append((os.path.basename(img_path), predicted.item()))

# Create a DataFrame and save to CSV
submission = pd.DataFrame(predictions, columns=["image_id", "label"])
submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")
