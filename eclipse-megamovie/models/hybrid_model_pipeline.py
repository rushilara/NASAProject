import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2
import math
import os
import pandas as pd

# Pretrained model path
MODEL_PATH = "./models/940_efficientnet_b1_20241126-212325.pth"

# Test directory and output CSV
TEST_DIR = "../images/test"
OUTPUT_CSV = "predictions_with_debug.csv"
DEBUG_IMAGES_DIR = "./debug_circles"  # Directory to save debug images

# Image processing constants
IMG_SIZE = (300, 300)  # Match training size
MIN_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for circle detection

# Load EfficientNet model
def load_model(model_path):
    model = EfficientNet.from_name('efficientnet-b1')  # Initialize EfficientNet
    model._fc = nn.Linear(model._fc.in_features, 8)   # 8 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize the model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Image preprocessing for EfficientNet
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Predict class using EfficientNet
def predict_class(image_path, model):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = torch.argmax(output).item()
    return predicted_class

# Circle overlap calculation
def calculate_overlap(r1, r2, d):
    if d >= r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2
    else:
        part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        return part1 + part2 - part3

# Classify partial eclipse with debug output and confidence thresholds
def classify_partial_eclipse(image, image_path=None, output_debug_dir=DEBUG_IMAGES_DIR):
    os.makedirs(output_debug_dir, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=300
    )

    # Debug: Save image with detected circles
    if circles is not None:
        confidence_scores = []
        for circle in circles[0, :]:
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            confidence_scores.append(radius / 300)  # Normalize confidence score
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw circle
            cv2.circle(image, center, 2, (255, 0, 0), 3)  # Draw center

        # Save debug image
        debug_path = os.path.join(output_debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, image)

        # Check confidence threshold
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        if avg_confidence < MIN_CONFIDENCE_THRESHOLD:
            return None  # Insufficient confidence, do not refine

    # Ensure at least 2 circles are detected
    if circles is not None and len(circles[0]) >= 2:
        x1, y1, r1 = circles[0][0]
        x2, y2, r2 = circles[0][1]
        d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        overlap_area = calculate_overlap(r1, r2, d)
        sun_area = math.pi * r1**2
        coverage_percentage = (overlap_area / sun_area) * 100

        # Return bin based on coverage percentage
        if coverage_percentage <= 25:
            return 1
        elif 26 <= coverage_percentage <= 55:
            return 2
        elif 56 <= coverage_percentage <= 99:
            return 3
    return None

# Hybrid classifier
def hybrid_classifier(image_path, model):
    predicted_class = predict_class(image_path, model)

    # Refine classification for ambiguous bins (0-95% partial eclipse)
    if predicted_class in [1, 2, 3]:  # 0-25%, 26-55%, 56-99%
        raw_image = cv2.imread(image_path)
        refined_class = classify_partial_eclipse(raw_image, image_path)
        if refined_class is not None:
            return refined_class

    return predicted_class

# Generate predictions and save debug images
def generate_predictions(test_dir, model, output_csv, debug_dir=DEBUG_IMAGES_DIR):
    predictions = []
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        label = hybrid_classifier(image_path, model)
        predictions.append({"image_id": image_name, "label": label})

    # Save predictions to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"Debug images saved to {debug_dir}")

# Run the pipeline
if __name__ == "__main__":
    generate_predictions(TEST_DIR, model, OUTPUT_CSV)
