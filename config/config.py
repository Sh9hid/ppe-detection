import os
from dotenv import load_dotenv

load_dotenv()


INPUT_DIR = os.getenv('INPUT_DIR')
OUTPUT_DIR_PERSON = os.getenv('OUTPUT_DIR_PERSON')
OUTPUT_DIR_PPE = os.getenv('OUTPUT_DIR_PPE')