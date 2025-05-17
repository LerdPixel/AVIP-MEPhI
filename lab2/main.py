import os
import numpy as np
import cv2

input_folder = './'
output_folder = './'

def load_image(image_name):
    return cv2.imread(os.path.join(input_folder, image_name))

def save_image(image, image_name):
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)


def to_grayscale(image):
    grayscale = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    return grayscale.astype(np.uint8)

def binarize_image(grayscale_image, threshold=128):
    binary_image = grayscale_image > threshold
    return binary_image.astype(np.uint8) * 255

def adaptive_binarization(image, window_size=3):
    grayscale_image = to_grayscale(image)
    output_image = np.zeros_like(grayscale_image)
    padding = window_size // 2
    for i in range(padding, grayscale_image.shape[0] - padding):
        for j in range(padding, grayscale_image.shape[1] - padding):
            window = grayscale_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            threshold = np.mean(window)
            output_image[i, j] = 255 if grayscale_image[i, j] > threshold else 0
    return output_image

def adaptive_binarization_bradley_roth(image_path, threshold=0.15):
    #Реализация адаптивной бинаризации Брэдли и Рота с окном 3x3.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    height, width = image.shape
    integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)
    
    output = np.zeros_like(image, dtype=np.uint8)
    half_window = 1  # Окно 3x3
    
    for y in range(height):
        for x in range(width):
            x1 = max(x - half_window, 0)
            y1 = max(y - half_window, 0)
            x2 = min(x + half_window, width - 1)
            y2 = min(y + half_window, height - 1)
            
            count = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            sum_pixels = (integral_image[y2, x2] -
                          (integral_image[y1 - 1, x2] if y1 > 0 else 0) -
                          (integral_image[y2, x1 - 1] if x1 > 0 else 0) +
                          (integral_image[y1 - 1, x1 - 1] if (y1 > 0 and x1 > 0) else 0))
            
            mean_intensity = sum_pixels / count
            output[y, x] = 255 if image[y, x] >= mean_intensity * (1 - threshold) else 0
    
    cv2.imwrite("binarized_" + image_path, output)
    return output
image_name = 'cart.png'

image = load_image(image_name)

grayscale_image = to_grayscale(image)
save_image(grayscale_image, 'grayscale_' + image_name)

binary_image = binarize_image(grayscale_image, threshold=150)
save_image(binary_image, 'binary_' + image_name)

adaptive_binarization_bradley_roth(image_name)
