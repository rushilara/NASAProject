import os
import shutil
import pandas as pd

# Paths
predictions_file = "predictions_with_debug.csv"  # Path to predictions CSV
test_dir = "../images/test"           # Directory containing test images
output_dir = "../images/updated_predicted"  # Output directory to store sorted images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
for i in range(8):  # Create subdirectories for bins 0-7
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

# Read predictions CSV
predictions = pd.read_csv(predictions_file)

# Sort images into bins
for _, row in predictions.iterrows():
    image_id = row['image_id']
    label = row['label']
    
    # Source image path
    src_path = os.path.join(test_dir, image_id)
    
    # Destination directory
    dest_dir = os.path.join(output_dir, str(label))
    dest_path = os.path.join(dest_dir, image_id)
    
    # Move or copy the image
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)  # Use copy2 to retain metadata
    else:
        print(f"Warning: {src_path} does not exist.")

print(f"Images have been sorted into bins in '{output_dir}'.")
