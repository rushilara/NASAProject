import os
import pandas as pd
import shutil 

# Load the CSV file
data = pd.read_csv('train.csv')

root_dir = 'grouped_images'

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

# Create a fresh root directory and subdirectories for each label (0 through 7)
os.makedirs(root_dir, exist_ok=True)
for label in range(8):
    os.makedirs(os.path.join(root_dir, str(label)), exist_ok=True)


for _, row in data.iterrows():
    image_id = row['image_id']
    label = str(row['label'])
    
    source_path = os.path.join('./train', image_id)
    destination_path = os.path.join(root_dir, label, image_id)
    
    shutil.copy(source_path, destination_path)



