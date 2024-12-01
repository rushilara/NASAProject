import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2
import math
import os
import pandas as pd
import numpy as np

# Pretrained model path
MODEL_PATH = "./models/best_model.pth"  # Update with your actual model path

# Test directory and output CSV
TEST_DIR = "../images/test"  # Update if necessary
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
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

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
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray_blurred, 50, 150)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=150
    )

    # Debug: Save image with detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        confidence_scores = []
        radii = []
        for circle in circles[0, :]:
            x, y, r = circle
            radii.append(r)
            # No direct confidence score from HoughCircles, so we can use the circle radius as a proxy
            # Alternatively, we can assume detection is confident if circles are detected
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Draw circle
            cv2.circle(image, (x, y), 2, (255, 0, 0), 3)  # Draw center

        # Save debug image
        debug_path = os.path.join(output_debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, image)

        # Proceed only if at least two circles are detected
        if len(circles[0]) >= 2:
            # Assume the largest circle is the sun, the next largest is the moon
            sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
            x1, y1, r1 = sorted_circles[0]
            x2, y2, r2 = sorted_circles[1]
            d = math.hypot(x1 - x2, y1 - y2)
            overlap_area = calculate_overlap(r1, r2, d)
            sun_area = math.pi * r1**2
            coverage_percentage = (overlap_area / sun_area) * 100

            # Return bin based on coverage percentage
            if coverage_percentage <= 25:
                return 1  # Bin for 0-25%
            elif 25 < coverage_percentage <= 55:
                return 2  # Bin for 26-55%
            elif 55 < coverage_percentage <= 95:
                return 3  # Bin for 56-95%
    return None  # Return None if verification is not possible

# Hybrid classifier
def hybrid_classifier(image_path, model):
    predicted_class, model_confidence = predict_class(image_path, model)

    # Only proceed if model confidence is below a threshold and predicted class is in bins 1-3
    if predicted_class in [1, 2, 3] and model_confidence < 0.9:
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print(f"Error loading image {image_path}")
            return predicted_class  # Return original prediction
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

