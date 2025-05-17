import cv2
import numpy as np
import os
from PIL import Image

def compute_gradients_sobel(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # ядра Собеля
    sobel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[ 1,  2,  1], 
                         [ 0,  0,  0], 
                         [-1, -2, -1]], dtype=np.float32)

    Gx = cv2.filter2D(gray, cv2.CV_64F, sobel_x)
    Gy = cv2.filter2D(gray, cv2.CV_64F, sobel_y)

    # G = |Gx| + |Gy|
    G = np.abs(Gx) + np.abs(Gy)

    Gx = cv2.normalize(np.abs(Gx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Gy = cv2.normalize(np.abs(Gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    G = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_gradient = cv2.threshold(G, 50, 255, cv2.THRESH_BINARY)

    base_name = os.path.splitext(image_file)[0]
    Image.fromarray(Gx).save(f'{base_name}_Gx.png')
    Image.fromarray(Gy).save(f'{base_name}_Gy.png')
    Image.fromarray(G).save(f'{base_name}_G.png')
    Image.fromarray(binary_gradient).save(f'{base_name}_bin.png')
    return Gx, Gy, G



input_folder = 'in'
output_folder = 'out'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    compute_gradients_sobel(image_path)