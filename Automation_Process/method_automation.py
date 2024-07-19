'''
1. Make sure to download poppler from here: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the downloaded folder
3. Place the folder in whichever directory you wish to
4. Navigate to the "bin" folder
5. Copy the path of the "bin" folder and add to PATH variables
'''

import os
import base64
import requests
from pdf2image import convert_from_path


# Function to convert PDF to JPG
def convert_pdf_to_jpg(input_dir, output_dir):
    """
    Converts all .pdf files in the input directory to .jpg images and saves them in the output directory.

    Arguments:
        input_dir (str): Path to the input directory containing PDF files.
        output_dir (str): Path to the output directory where converted JPG images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)  # Get the full path of the PDF file
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.jpg')  # Construct the output image path

            try:
                images = convert_from_path(pdf_path)
                for i, img in enumerate(images):
                    img_path = f'{output_path[:-4]}_{i}.jpg' if i > 0 else output_path  # Add index to output image path if multiple pages
                    img.save(img_path, 'JPEG')

            except Exception as e:
                print(f'Error converting {pdf_path}: {e}')

            else:
                print(f'Successfully converted {pdf_path} to {output_path}')


# Specifying Input and Output Directories
input_dir = r"Automation_Process\\Purchase_Orders_PDF"
output_dir = r"Automation_Process\\Purchase_Orders_JPG"

convert_pdf_to_jpg(input_dir, output_dir)


# Specifying required variables for label-studio
# URL Bar of the browser: http://localhost:8080/projects/3/data?tab=3
folder_path = r'Automation_Process\\Purchase_Orders_JPG' # Set accordingly
project_id = '3' # Set accordingly | can be found in the url bar of the browser
api_key = '8264892ad85dc086e82b4a1168e5174ccbfd63fb' # Set accordingly
label_studio_url = 'http://localhost:8080' # Set accordingly | can be found in the url bar of the browser


# Function to upload an image to label-studio
def upload_image_to_label_studio(image_path, project_id, api_key, label_studio_url):
    # Read the image file
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Prepare the data for the API request
    data = {
        "data": {
            "image": f"data:image/jpeg;base64,{encoded_string}"
        }
    }

    # Set up the headers
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    # Make the API request
    url = f"{label_studio_url}/api/projects/{project_id}/import"
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 201:
        print(f"Successfully uploaded {os.path.basename(image_path)}")
    else:
        print(f"Failed to upload {os.path.basename(image_path)}. Status code: {response.status_code}")
        print(f"Response: {response.text}")


# Function to iterate through the output directory to upload each individual image
def upload_images_from_folder(folder_path, project_id, api_key, label_studio_url):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            upload_image_to_label_studio(image_path, project_id, api_key, label_studio_url)


upload_images_from_folder(folder_path, project_id, api_key, label_studio_url)