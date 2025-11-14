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

def main():
    
    source_root = "docs-sm"
    output_root = "text-data"
    
    os.makedirs(output_root, exist_ok=True)
    
    # get all category folders (advertisement, budget, email, etc.)
    category_folders = list_folders(source_root)
    
    # process each category folder
    for category in category_folders:
        # create corresponding output folder
        output_category_path = os.path.join(output_root, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        # get all image files in this category
        category_path = os.path.join(source_root, category)
        image_files = get_folder_files(category_path)
        
        # process each image file
        for image_path in image_files:
            # get filename without extension (ex "0000022394" from "0000022394.jpg")
            filename = os.path.basename(image_path)
            file_id = os.path.splitext(filename)[0]
            
            # make output txt file path
            output_text_path = os.path.join(output_category_path, f"{file_id}.txt")
            
            # check if already processed
            if os.path.exists(output_text_path):
                continue
            
            # extract text from image using google/gemma-3-12b vision model
            extracted_text = process_image_llm(image_path)
                
            # save text to file
            with open(output_text_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(extracted_text)
                    
            
if __name__ == "__main__":
    main()