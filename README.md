
# Person and PPE Detection (You-Only-Look-Once v8)
Trained on person and ppe-class. utils\preview.py for viewing most-inaccurate labels. 

Here's the link to the [Google Colab notebook](https://colab.research.google.com/drive/1Wrp33aq26xWiuC8pJQwP4IQxoXzcmoee?usp=sharing).

 ## Installation

 1. Create a new environment

```
python -m venv .venv
.venv/Scripts/activate # on windows
or
source .venv/bin/activate # on linux
```
2. Create a new .env file and edit file paths.

```
# Define the directories
INPUT_DIR = 'datasets\labels'
OUTPUT_DIR_PERSON = 'datasets/labels-person' 
OUTPUT_DIR_PPE = 'datasets/labels-ppe'
```

3. Install Dependencies

```
pip install -r requirements.txt
```

4. Perform inference using the cli using,

```
yolo predict model='models\best.pt' source=' path/to/jpeg/dir'
```
or using opencv bounding boxes

```
python perform_inference.py path/to/input_dir, path/to/output_dir, path/to/person_det_model path/to/ppe_detection_model
```

## Summary: 

1. Utils contains split_data.py for training and preview.py to view bounding boxes to drop the most inaccurate classes.
2. Models in weights directory. 
3. Trained on 20 epochs each for persons and ppe. 
4. perform_inference.py and pascalVOC_to_yolo.py using argparse for inference using the cli.
5. predict-cv, predict-person, predict-ppe for bounding boxes with open-cv, model1 i.e person class, model2 i.e ppe-class.


## ROADMAP

1. Improve accuracy.
2. Drop hard-hat, mask, etc. Train with fewer labels.
3. More readability.
4. More modularity, better code-structure.
5. Add summary to notebook.  
6. Dockerize.
