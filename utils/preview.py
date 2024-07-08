import os
import cv2

def draw_multiple_bounding_boxes(labels_folder_path, images_folder_path):
    """
    This function takes in a labels folder in YOLO format and
    for previewing multiple bounding boxes on the corresponding image.
    """
    
    # Define the class names
    class_names = ['glasses', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness'] # originally trained on more classnames

    # Function to parse YOLO label files
    def parse_yolo_label(label_file_path, image_width, image_height):
        boxes = []
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                label = class_names[class_id]
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                boxes.append((x1, y1, x2, y2, label))
        return boxes

    # Process each image and its corresponding label file
    for image_name in os.listdir(images_folder_path):
        image_path = os.path.join(images_folder_path, image_name)
        label_path = os.path.join(labels_folder_path, os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        
        # Parse the YOLO label file to get bounding boxes
        boxes = parse_yolo_label(label_path, image_width, image_height)
        
        # Draw the bounding boxes
        for x1, y1, x2, y2, label in boxes:
            color = (0, 255, 0)  # Green color for the bounding box
            thickness = 2  # Thickness of the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Call the function
draw_multiple_bounding_boxes('datasets/labels-ppe', 'dataset/crops/person')