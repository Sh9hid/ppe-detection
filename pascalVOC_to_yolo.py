import os
import xml.etree.ElementTree as ET
from config.config import INPUT_DIR, OUTPUT_DIR_PERSON, OUTPUT_DIR_PPE
import argparse

def xml_to_yolo(size, box):
    """
    Convert XML bounding box coordinates to YOLO format.

    Args:
        size (tuple): Image size (width, height)
        box (tuple): XML bounding box coordinates (xmin, xmax, ymin, ymax)

    Returns:
        tuple: YOLO bounding box coordinates (x, y, w, h)
    """
    dw = 1.0 / size
    dh = 1.0 / size
    x = (box + box) / 2.0 - 1
    y = (box + box) / 2.0 - 1
    w = box - box
    h = box - box
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_to_yolo(input_dir, output_dir_person, output_dir_ppe):
    """
    Convert XML annotations to YOLO format.

    Args:
        input_dir (str): Input directory containing XML files
        output_dir_person (str): Output directory for 'person' annotations
        output_dir_ppe (str): Output directory for 'ppe' annotations
    """
    ppe_classes = ['hard-hat', 'gloves', 'mask', 'boots', 'glasses', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']
    class_mapping = {cls: idx for idx, cls in enumerate(ppe_classes)}

    for filename in os.listdir(input_dir):
        if not filename.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_annotations = []
        
        for obj in root.findall('object'):
            difficult = obj.find('difficult')
            if difficult is not None and difficult.text == '1':
                continue
            cls = obj.find('name').text

            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = xml_to_yolo((width, height), b)

            class_id = class_mapping.get(cls)

            if cls.lower() != 'person':
                if class_id is not None:
                    yolo_annotations.append(f"{class_id} {' '.join([str(a) for a in bb])}")
                    
                yolo_filename = os.path.join(output_dir_ppe, f"{os.path.splitext(filename)}.txt")
                with open(yolo_filename, 'w') as f:
                    for annotation in yolo_annotations:
                        f.write(f"{annotation}\n")
            else:
                if class_id is not None:
                    yolo_annotations.append(f"0 {' '.join([str(a) for a in bb])}")
        
        yolo_filename = os.path.join(output_dir_person, f"{os.path.splitext(filename)}.txt")
        with open(yolo_filename, 'w') as f:
            for annotation in yolo_annotations:
                f.write(f"{annotation}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format.')
    parser.add_argument('input_dir', type=str, help='Input directory containing XML files.')
    parser.add_argument('output_dir_person', type=str, help='Output directory for person annotations.')
    parser.add_argument('output_dir_ppe', type=str, help='Output directory for PPE annotations.')

    args = parser.parse_args()

    os.makedirs(args.output_dir_person, exist_ok=True)
    os.makedirs(args.output_dir_ppe, exist_ok=True)

    convert_to_yolo(args.input_dir, args.output_dir_person, args.output_dir_ppe)

    print("Conversion complete!")