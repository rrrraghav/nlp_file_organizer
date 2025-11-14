# we want to do this so we can use sklearn:
"""
load_files from sklearn using folder structure like this:

text-data/
├─ advertisement/
│  ├─ 123.txt
│  ├─ 234.txt
├─ email/
│  ├─ 345.txt
│  ├─ 456.txt

Each subfolder name becomes a class label and 
each file inside the folder is treated as a sample/document.

Reads all the files and returns an object with:
- data (list of str text content of each file, can vectorize)
- target (list of integer labels (0, 1, 2…) for each class)
- target_names (list of class names which is subfolders)
- filenames (list of paths to each file, optional)
"""

"""
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = load_files('text-data', encoding='utf-8', decode_error='ignore')

X = data.data         
y = data.target       
class_names = data.target_names

# should be ex ['advertisement', 'email', 'invoice']
print(class_names)      

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectors, y)

tweak parameters with grid search or random search
once preprocessing done
"""