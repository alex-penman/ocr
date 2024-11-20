from google.cloud import vision
from lets_translate import convert_from_path
import io
import os

# File paths
pdf_file = "studentText1.pdf"
output_text_file = "ocr_output.txt"

# Initialize the Google Vision client
client = vision.ImageAnnotatorClient()

def detect_text(image_path):
    """Uses Google Vision API to detect text in an image."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    
    # Run OCR on the image
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f"Error during text detection: {response.error.message}")

    # Return extracted text
    return response.full_text_annotation.text if response.full_text_annotation else ""

# Convert PDF to images
pages = convert_from_path(pdf_file, 300)
ocr_output = []

# Process each page and perform OCR
for i, page in enumerate(pages, start=1):
    image_path = f'page_{i}.png'
    page.save(image_path, 'PNG')  # Save each page as an image
    text = detect_text(image_path)  # Extract text from the image
    ocr_output.append(f"Page {i}:\n{text}\n{'-'*40}\n")

# Save OCR results to a text file
with open(output_text_file, "w") as file:
    file.write("\n".join(ocr_output))

print("OCR completed. Text saved to ocr_output.txt.")
