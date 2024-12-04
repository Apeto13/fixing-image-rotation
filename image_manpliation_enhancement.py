import base64
import cv2
import math
import numpy as np
import os
from deskew import determine_skew
from typing import List, Tuple, Union, Optional
import tempfile
from pdf2image import convert_from_path


def delete_all_white_outlines(image: np.ndarray, background_color: Tuple[int, int, int] = (255, 255, 255), threshold: int = 200) -> np.ndarray:
    """
    Deletes all white outlines around the main content of an image by cropping to the content area.

    Parameters:
    - image: np.ndarray - Input image from which white outlines are to be removed.
    - background_color: Tuple[int, int, int] - The RGB color of the background to be considered as white outline (default is white).
    - threshold: int - Threshold to consider a pixel as white (default is 200).

    Returns:
    - np.ndarray - Cropped image without white borders.
    """
    # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image (content vs background)
    _, binary = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

    # Invert binary image to make content white and background black
    binary_inverted = cv2.bitwise_not(binary)

    # Find contours around main content
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return image  # Return original if no contours are found

    # Find bounding box for the largest contour (main content area)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Optional: Add margin around the content to avoid tight cropping
    margin = 10  # Adjust margin size as needed
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, image.shape[1] - x)
    h = min(h + 2 * margin, image.shape[0] - y)

    # Crop the image to the bounding box with margin
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image

def crop_white_spaces_temp(image: np.ndarray, threshold=200) -> np.ndarray:
    """
    Crops empty (white) parts from all sides of an image.
    
    Args:
        image: The image to crop.
        threshold: Pixel value threshold to consider as white (0-255, default: 250)
    
    Returns:
        Cropped image without white borders.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Get image dimensions
    height, width = gray.shape

    # Find the first non-white row from the top
    top = 0
    for i in range(height):
        if np.mean(gray[i]) < threshold:
            top = i
            break

    # Find the first non-white row from the bottom
    bottom = height - 1
    for i in range(height - 1, -1, -1):
        if np.mean(gray[i]) < threshold:
            bottom = i
            break

    # Find the first non-white column from the left
    left = 0
    for i in range(width):
        if np.mean(gray[:, i]) < threshold:
            left = i
            break

    # Find the first non-white column from the right
    right = width - 1
    for i in range(width - 1, -1, -1):
        if np.mean(gray[:, i]) < threshold:
            right = i
            break

    # Add padding
    padding = 10
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)

    # Crop the original image
    cropped = image[top:bottom+1, left:right+1]
    
    return cropped

def _rotate_photo(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    """Rotates an image by a specified angle and fills empty areas with the background color."""
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    new_width = int(abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width))
    new_height = int(abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height))

    image_center = (old_width / 2, old_height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[0, 2] += (new_width / 2) - image_center[0]
    rot_mat[1, 2] += (new_height / 2) - image_center[1]

    rotated_image = cv2.warpAffine(image, rot_mat, (new_width, new_height), borderValue=background)
    return rotated_image

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhances the image by increasing contrast and sharpness."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced

def double_image_size(image: np.ndarray) -> np.ndarray:
    """Doubles the size of the image using INTER_CUBIC interpolation."""
    height, width = image.shape[:2]
    new_size = (width * 2, height * 2)
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return enlarged_image

import os
import cv2
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path: str, output_folder: str) -> list:
    """
    Converts a PDF file into images, saving each page as an image in the specified folder.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to save the output images.

    Returns:
        list: A list of file paths for the converted images.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    try:
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i + 1}.jpg"
            image_path = os.path.join(output_folder, image_name)
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
    except Exception as e:
        print(f"Error converting PDF '{pdf_path}': {e}")

    return image_paths

def deskew_photos(input_folder: str, base_output_folder: str):
    """
    Deskews and enhances all images in the input folder (including subdirectories) 
    and saves them directly into a single output folder.
    """
    # Ensure the base output folder exists
    os.makedirs(base_output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)

                # Convert PDF to images
                image_paths = convert_pdf_to_images(pdf_path, base_output_folder)

                # Process each image generated from the PDF
                for image_path in image_paths:
                    process_image(image_path, base_output_folder)

            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                process_image(image_path, base_output_folder)


def process_image(input_path: str, output_folder: str):
    """
    Processes an individual image: deskew, enhance, and crop.

    Args:
        input_path (str): Path to the input image.
        output_folder (str): Folder to save the processed image.
    """
    output_path = os.path.join(output_folder, os.path.basename(input_path))

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image '{input_path}'. Skipping...")
        return

    scale_percent = 125
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale and determine skew
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    print(f"This is the angle of the image:{angle}")

    rotated = _rotate_photo(image, angle, (255, 255, 255))

    enhanced_image = enhance_image(rotated)

    final_image = crop_white_spaces_temp(enhanced_image, threshold=250)

    cv2.imwrite(output_path, final_image)
    print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # Specify the folder name to process
    folderName = "files"

    # Set the input and output folders
    input_folder = os.path.join("before", folderName)
    base_output_folder = os.path.join("after", folderName)

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
    else:
        print(f"Processing files from '{input_folder}' to '{base_output_folder}'...")
        deskew_photos(input_folder, base_output_folder)
        print("Processing complete.")