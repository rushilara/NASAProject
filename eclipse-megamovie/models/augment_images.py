import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from shutil import copy2
import gc

# Configuration
input_dir = "../images/enhanced_grouped_images"
output_dir = "../images/balanced_training"
target_counts = {
    0: 2500,
    1: 1500,
    2: 1500,
    3: 2000,
    4: 1500,
    5: 2000,
    6: 500,
    7: 1000
}

# Define data augmentation parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=17,
    width_shift_range=0.1,
    height_shift_range=0.07,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=10,
    channel_shift_range=80,
    fill_mode="nearest"
)

# Ensure output directory structure exists
os.makedirs(output_dir, exist_ok=True)
for category in target_counts.keys():
    os.makedirs(os.path.join(output_dir, str(category)), exist_ok=True)

# Augment each category
for category, target_count in target_counts.items():
    category_dir = os.path.join(input_dir, str(category))
    output_category_dir = os.path.join(output_dir, str(category))

    if not os.path.exists(category_dir):
        print(f"Category directory {category_dir} does not exist. Skipping...")
        continue

    images = [img for img in os.listdir(category_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    print(f"Processing category {category} with {len(images)} images.")

    current_count = len(images)
    augment_count = target_count - current_count

    if augment_count <= 0:
        print(f"No augmentation needed for category {category}.")
        for img_name in images:
            copy2(os.path.join(category_dir, img_name), output_category_dir)
        continue

    print(f"Generating {augment_count} augmented images for category {category}.")
    images_to_augment = images * (augment_count // len(images)) + images[:augment_count % len(images)]

    for img_name in images_to_augment:
        img_path = os.path.join(category_dir, img_name)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        augmented_images = datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=output_category_dir,
            save_prefix="aug",
            save_format="jpg"
        )

        # Generate one augmented image per loop iteration
        next(augmented_images)

        # Clean up memory
        del img, img_array
        gc.collect()

    print(f"Finished category {category}.")

print(f"Data augmentation completed. Augmented images saved to '{output_dir}'.")
