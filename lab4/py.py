import numpy as np
from PIL import Image

def load_grayscale_image(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image, dtype=np.float32)

def apply_convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image)
    # Применяем свёртку
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+kh, j:j+kw]  
            result[i, j] = np.sum(region * kernel) 
    return result

def binary_threshold(image, threshold=100):
    """Бинаризует изображение по порогу"""
    binary = (image > threshold) * 255  # Все, что выше порога — белое, остальное — черное
    return binary.astype(np.uint8)

def compute_gradients_sobel(image_path):
    image = load_grayscale_image(image_path)
    # ядра Собеля
    sobel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[ 1,  2,  1], 
                         [ 0,  0,  0], 
                         [-1, -2, -1]], dtype=np.float32)
    # фильтры Собеля
    Gx = apply_convolution(image, sobel_x)
    Gy = apply_convolution(image, sobel_y)
    # G = |Gx| + |Gy|
    G = np.abs(Gx) + np.abs(Gy)
    G_binary = binary_threshold(G, threshold)
    Gx = ((Gx - Gx.min()) / (Gx.max() - Gx.min()) * 255).astype(np.uint8)
    Gy = ((Gy - Gy.min()) / (Gy.max() - Gy.min()) * 255).astype(np.uint8)
    G = ((G - G.min()) / (G.max() - G.min()) * 255).astype(np.uint8)
    Image.fromarray(Gx).save("Gx.png")
    Image.fromarray(Gy).save("Gy.png")
    Image.fromarray(G).save("G.png")
    Image.fromarray(G_binary).save("binG.png")
    return Gx, Gy, G
image_path = "input.png" 
compute_gradients_sobel(image_path)