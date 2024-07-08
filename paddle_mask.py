import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import argparse
from tqdm import tqdm
import logging
import multiprocessing
import subprocess

def get_num_gpus():
    """Returns the number of available GPUs."""
    result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    return len(result.stdout.decode('utf-8').strip().split('\n'))

def process_images(child_input_dir, child_output_dir, gpu_index):
    # Set the visible GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Initialize PaddleOCR without angle classification and with Chinese language
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=True, det_db_score_mode="slow", use_dilation=True)

    # Ensure the output directory exists
    os.makedirs(child_output_dir, exist_ok=True)

    # Loop through all files in the input directory with progress bar
    for filename in tqdm(os.listdir(child_input_dir), desc=f"Processing {child_input_dir}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct full file path
            image_path = os.path.join(child_input_dir, filename)
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read {image_path}")
                continue

            # Perform OCR to detect text regions
            result = ocr.ocr(image, cls=False)

            # Create a black image with the same dimensions as the original
            black_image = np.zeros_like(image)

            if result and result != [None]:
                # Fill detected text regions with white color
                for line in result:
                    for word_info in line:
                        bbox = word_info[0]
                        bbox = np.array(bbox).astype(np.int32)
                        cv2.fillPoly(black_image, [bbox], color=(255, 255, 255))

            # Save the result
            output_path = os.path.join(child_output_dir, filename)
            cv2.imwrite(output_path, black_image)

def process_child_folder(args):
    child_input_dir, child_output_dir, gpu_index = args
    process_images(child_input_dir, child_output_dir, gpu_index)

def process_parent_folder(parent_input_dir, parent_output_dir):
    # Check if the parent input directory exists
    if not os.path.exists(parent_input_dir):
        print(f"Parent input directory '{parent_input_dir}' does not exist.")
        return
    
    # Get the number of GPUs
    num_gpus = get_num_gpus()

    # Create a list of arguments for each child folder
    tasks = []
    child_folders = [f for f in os.listdir(parent_input_dir) if os.path.isdir(os.path.join(parent_input_dir, f))]
    for i, child_folder_name in enumerate(child_folders):
        child_input_dir = os.path.join(parent_input_dir, child_folder_name)
        child_output_dir = os.path.join(parent_output_dir, child_folder_name)
        gpu_index = i % num_gpus
        tasks.append((child_input_dir, child_output_dir, gpu_index))

    # Use multiprocessing to process each child folder in parallel
    with multiprocessing.Pool(processes=num_gpus) as pool:
        pool.map(process_child_folder, tasks)

def main():
    parser = argparse.ArgumentParser(description="Process images in multiple child folders within a parent folder.")
    parser.add_argument('--parent-input-dir', type=str, required=True, help='The parent directory containing child input folders.')
    parser.add_argument('--parent-output-dir', type=str, required=True, help='The parent directory to save child output folders.')

    args = parser.parse_args()
    
    process_parent_folder(args.parent_input_dir, args.parent_output_dir)

if __name__ == "__main__":
    # Suppress specific debug prints from paddleocr
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    
    main()
