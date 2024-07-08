import os
import shutil
from sklearn.model_selection import train_test_split

# Define directories
images_dir = 'dataset\crops\person'  # Directory containing images
labels_dir = 'datasets\labels-ppe'  # Directory containing YOLO format label files
output_dir = 'images'  # Output directory to store train/val splits

# Create output directories if they don't exist
train_images_dir = os.path.join(output_dir, 'images/train')
val_images_dir = os.path.join(output_dir, 'images/val')
train_labels_dir = os.path.join(output_dir, 'labels/train')
val_labels_dir = os.path.join(output_dir, 'labels/val')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Split the files into training and validation sets
train_files, val_files = train_test_split(image_files, train_size=0.8, random_state=42)

def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"Warning: File not found - {src_file}")

# Copy training images and labels
copy_files(train_files, images_dir, train_images_dir)
copy_files([f.replace('.jpg', '.txt').replace('.png', '.txt') for f in train_files], labels_dir, train_labels_dir)

# Copy validation images and labels
copy_files(val_files, images_dir, val_images_dir)
copy_files([f.replace('.jpg', '.txt').replace('.png', '.txt') for f in val_files], labels_dir, val_labels_dir)

