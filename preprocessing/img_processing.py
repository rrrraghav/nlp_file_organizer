import os
import base64
import asyncio
import aiohttp

BASE_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-3-12b"

def list_folders(folder_path: str):
    """
    Get all folders in a given folder

    Args:
        folder_path (str): Path to folder with folders

    Returns:
        list: List of folder names (not full paths)
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
    Get all files in a given folder

    Args:
        folder_path (str): Path to folder with files

    Returns:
        list: List of full file paths
    """
    file_list = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            file_list.append(full_path)
    return file_list

async def process_image_llm(session, image_path, semaphore):
    """
    Extract text from an image using googles gemma-3-12b through LM Studios API
    
    Args:
        session (aiohttp.ClientSession): Shared HTTP session for async requests
        image_path (str): Path to the image file
        semaphore (asyncio.Semaphore): Controls concurrent request limit
    
    Returns:
        str: Extracted text from the image
    """
    # Semaphore limits how many concurrent API requests we make to LM Studio
    # to prevent exploding GPU with too many simultaneous inference requests
    # which could cause oom errors or slow down processing
    async with semaphore:
        # read and encode image to base64
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
            # only one pagers so generous limit
            "max_tokens": 2000
        }
        
        async with session.post(BASE_URL, json=payload) as response:
            result = await response.json()
            return result['choices'][0]['message']['content']
    
async def process_single_image(session, image_path, output_text_path, semaphore):
    """
    Process a single image and save extracted text to txt file
    
    Args:
        session (aiohttp.ClientSession): Shared HTTP session for async requests
        image_path (str): Path to the input image file
        output_text_path (str): Path where the extracted text will be saved
        semaphore (asyncio.Semaphore): Controls concurrent request limit
    """
    # check if already processed (skip to avoid re-processing)
    if os.path.exists(output_text_path):
        return
    
    # extract text using vision model
    extracted_text = await process_image_llm(session, image_path, semaphore)
    
    if extracted_text:
        with open(output_text_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)

async def main():
    """
    Process all images in category folders
    """
    source_root = "docs-sm"
    output_root = "text-data"
    
    # config for how many images are processed simultaneously
    # semaphore make sure we never exceed this limit
    MAX_CONCURRENT_REQUESTS = 4
    
    os.makedirs(output_root, exist_ok=True)

    category_folders = list_folders(source_root)
    
    tasks = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # create aiohttp session (reuse for all requests)
    async with aiohttp.ClientSession() as session:
        for category in category_folders:
            # make output folder for this category
            output_category_path = os.path.join(output_root, category)
            os.makedirs(output_category_path, exist_ok=True)
            
            # get all image files in this category
            category_path = os.path.join(source_root, category)
            image_files = get_folder_files(category_path)
            
            # create async tasks for each image file
            for image_path in image_files:
                filename = os.path.basename(image_path)
                file_id = os.path.splitext(filename)[0]
                output_text_path = os.path.join(output_category_path, f"{file_id}.txt")
                
                task = process_single_image(session, image_path, output_text_path, semaphore)
                tasks.append(task)
        
        # run all tasks concurrently (asyncio will manage the semaphore limits)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())