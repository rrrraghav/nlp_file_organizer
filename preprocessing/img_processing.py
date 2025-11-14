import os
import requests
import base64

def list_folders(folder_path: str):
    """
    get all folders in a given folder

    Args:
        folder_path (str): path to folder with folders

    Returns:
        list: list of paths to folders
    """
    folders = []
    all_entries = os.listdir(folder_path)

    # go through the dirs and check if they are directories
    for entry in all_entries:
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            folders.append(entry)
    return folders

def get_folder_files(folder_path: str):
    """
    get all files in a given folder

    Args:
        folder_path (str): path to folder with files

    Returns:
        list: list of files in folder
    """

    file_list = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            file_list.append(full_path)
    return file_list

# docs-sm for list_folders and result is ['adv', 'email'] etc
# for each folder we need to run get_folder_files with prepended docs-sm/... -> file paths

# here

# for each file in the folder prepend with docs-sm and folder name to then pass to image processing
# create a .txt file for of output from llm 
# .txt file needs to be stored in text-data/original_folder/same_id.txt

# here
BASE_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-3-12b"

def process_image_llm(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    prompt = "Extract all text from this image exactly as it appears. Output only the raw text with no formatting, " \
             "no explanations, no markdown, no bullet points. Preserve the original spacing and line breaks. " \
             "Pay close attention to spelling. Transcribe every word character-by-character exactly as shown, " \
             "including unusual spellings or names. Do not autocorrect or fix anything."
    
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        "max_tokens": 1000
    }
    
    response = requests.post(BASE_URL, json=payload)
    return response.json()['choices'][0]['message']['content']

# we want to do this so we can use sklearn:
"""
load_files from sklearn:

Expects a folder structure like this:

dataset_root/
├─ class_1/
│  ├─ file1.txt
│  ├─ file2.txt
├─ class_2/
│  ├─ file3.txt
│  ├─ file4.txt

Each subfolder name becomes a class label.

Each file inside the folder is treated as a sample/document.

Reads all the files and returns a Python object containing:
- data: list of strings (text content of each file) → ready for vectorization.
- target: list of integer labels (0, 1, 2…) corresponding to the class of each file.
- target_names: list of class names (subfolder names).
- filenames: list of paths to each file (optional, useful for debugging).
"""

"""
from sklearn.datasets import load_files

# Load data from folder structure
data = load_files('text-data', encoding='utf-8', decode_error='ignore')

# Access content
X = data.data           # list of text documents
y = data.target         # list of labels as integers
class_names = data.target_names  # list of class names
print(class_names)      # e.g., ['advertisement', 'email', 'invoice']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectors, y)
"""