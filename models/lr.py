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