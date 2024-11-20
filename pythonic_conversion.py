import io
import cv2
from pdf2image import convert_from_path
from google.cloud import vision
import easyocr
import os

# Initialize the Google Vision API and EasyOCR clients
client = vision.ImageAnnotatorClient()
easyocr_reader = easyocr.Reader(['fr'])  # French language support in EasyOCR

# Paths
pdf_file = "studentText1.pdf"
output_text_file = "ocr_output.txt"

# Function to preprocess images
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to increase contrast and remove noise
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Function for OCR with Google Vision
def google_vision_ocr(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # Specify French as the language hint
    image_context = vision.ImageContext(language_hints=["fr"])
    # Run OCR
    response = client.document_text_detection(image=image, image_context=image_context)
    if response.error.message:
        raise Exception(f"Google Vision OCR error: {response.error.message}")
    return response.full_text_annotation.text if response.full_text_annotation else ""

# Function for OCR with EasyOCR
def easyocr_ocr(image_path):
    result = easyocr_reader.readtext(image_path, detail=0)
    return " ".join(result)  # Join text fragments into a single string

# Convert PDF to images
pages = convert_from_path(pdf_file, 300)
ocr_output = []

# Process each page
for i, page in enumerate(pages, start=1):
    # Convert the PDF page to an OpenCV image
    image = cv2.cvtColor(page, cv2.COLOR_RGB2BGR)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Save preprocessed image for OCR
    image_path = f'page_{i}.png'
    cv2.imwrite(image_path, preprocessed_image)

    # Perform OCR with both Google Vision and EasyOCR
    google_result = google_vision_ocr(image_path)
    easyocr_result = easyocr_ocr(image_path)

    # Select the best result (you might compare or keep both for review)
    # For simplicity, we'll include both in the output
    ocr_output.append(f"Page {i} - Google Vision OCR:\n{google_result}\n{'-'*20}")
    ocr_output.append(f"Page {i} - EasyOCR:\n{easyocr_result}\n{'='*40}\n")

    # Optionally, delete the image after processing to save space
    os.remove(image_path)

# Save OCR results to a text file
with open(output_text_file, "w") as file:
    file.write("\n".join(ocr_output))

print("OCR completed. Text saved to ocr_output.txt.")
