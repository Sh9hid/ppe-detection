import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import argparse


# Step 1: Load models
def load_models(model1_path, model2_path):
    """
    Load the two YOLO models from the given paths.
    """
    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)
    return model1, model2

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

def define_class_names():
    """
    Define the class names for each model.
    TODO: drop incorrect classnames, like hard-hat ,mask, 
    etc improve accuracy
    """
    class_names1 = ['person']  
    class_names2 = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']
    return class_names1, class_names2

def process_image(image_path, model1, model2, class_names1, class_names2):
    """
    Process a single image by performing inference with both models and drawing bounding boxes.
    """
    image = cv2.imread(image_path)
    
    # Perform inference with both models
    results1 = model1(image)
    results2 = model2(image)

    # Draw bounding boxes for model1 results
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf)
            cls = int(box.cls)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put class name and confidence
            label = f'{class_names1[cls]} {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes for model2 results
    for r in results2:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Put class name and confidence
            label = f'{class_names2} {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

def save_image(image, output_path):
    """
    Save the image with bounding boxes to the given output path.
    """
    cv2.imwrite(output_path, image)

def perform_inference(input_dir, output_dir, model1_path, model2_path):
    """
    Perform object detection on all images in the input directory using the two models.
    """
    # Load the models
    model1, model2 = load_models(model1_path, model2_path)
    
    # Create the output directory
    create_output_directory(output_dir)
    
    # Define the class names
    class_names1, class_names2 = define_class_names()

    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            
            # Process the image
            image = process_image(image_path, model1, model2, class_names1, class_names2)
            
            # Save the image with bounding boxes
            output_path = os.path.join(output_dir, f'inference_{image_name}')
            save_image(image, output_path)
            
            print(f"Processed {image_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform object detection using two models.')
    parser.add_argument('input_dir', type=str, help='Input directory containing images.')
    parser.add_argument('output_dir', type=str, help='Output directory for saving results.')
    parser.add_argument('person_det_model', type=str, help='Path to the person detection model.')
    parser.add_argument('ppe_detection_model', type=str, help='Path to the PPE detection model.')

    args = parser.parse_args()

    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)