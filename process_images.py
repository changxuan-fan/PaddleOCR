import sys
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm
import logging

def process_images(child_input_dir, child_output_dir):
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=True, det_db_score_mode="slow", use_dilation=True)
    os.makedirs(child_output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(child_input_dir), desc=f"Processing {child_input_dir}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(child_input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read {image_path}")
                continue

            result = ocr.ocr(image, cls=False)
            black_image = np.zeros_like(image)

            if result and result != [None]:
                for line in result:
                    for word_info in line:
                        bbox = word_info[0]
                        bbox = np.array(bbox).astype(np.int32)
                        cv2.fillPoly(black_image, [bbox], color=(255, 255, 255))

            output_path = os.path.join(child_output_dir, filename)
            cv2.imwrite(output_path, black_image)

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_images.py <input_dir> <output_dir>")
        sys.exit(1)

    child_input_dir = sys.argv[1]
    child_output_dir = sys.argv[2]
    process_images(child_input_dir, child_output_dir)

if __name__ == "__main__":
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    main()
