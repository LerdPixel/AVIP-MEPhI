import numpy as np
import cv2
from PIL import Image

def from_np(image_array, filename):
    image = Image.fromarray(image_array)
    image.save(filename)
    
def save_image(image_array, filename):
    image = Image.fromarray(image_array)
    image.save(filename)

def split_rgb(image_path):
    """Разделяет каналы R, G, B, сохраняя их с исходными цветами"""
    image = Image.open(image_path).convert("RGB")
    r, g, b = image.split()
    
    r_image = Image.merge("RGB", (r, Image.new("L", r.size, 0), Image.new("L", r.size, 0)))
    g_image = Image.merge("RGB", (Image.new("L", g.size, 0), g, Image.new("L", g.size, 0)))
    b_image = Image.merge("RGB", (Image.new("L", b.size, 0), Image.new("L", b.size, 0), b))
    
    save_image(np.array(r_image), "R_channel.png")
    save_image(np.array(g_image), "G_channel.png")
    save_image(np.array(b_image), "B_channel.png")

def rgb_to_hsi(image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float32) / 255.0
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    intensity = (r + g + b) / 3.0
    intensity = (intensity * 255).astype(np.uint8)
    from_np(intensity, "HSI_intensity.png")

def invert_intensity1(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    intensity = (r + g + b) / 3.0
    
    inverted_intensity = 1.0 - intensity
    
    inverted_intensity = (inverted_intensity * 255).astype(np.uint8)
    from_np(inverted_intensity, "inverted_intensity.png")
def invert_intensity(image_path):
    """Инвертирует яркостную компоненту и сохраняет изображение"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    intensity = (r + g + b) / 3.0
    
    inverted_intensity = 1.0 - intensity  # Инверсия
    
    # Масштабируем в 8-битное изображение
    inverted_intensity = (inverted_intensity * 255).astype(np.uint8)
    from_np(inverted_intensity, "Inverted_intensity.png")

def resample_image(image_path, scale_x, scale_y, method, filename):
    """Передискретизация изображения с разными масштабами по горизонтали и вертикали"""
    image = Image.open(image_path)
    new_size = (int(image.width * scale_x), int(image.height * scale_y))
    resampled_image = image.resize(new_size, method)
    resampled_image.save(filename)

def stretch_image_x(image_path, Mx):
    """Растяжение изображения только по горизонтали"""
    resample_image(image_path, Mx, 1, Image.BILINEAR, "stretched_x.png")

def stretch_image_y(image_path, My):
    """Растяжение изображения только по вертикали"""
    resample_image(image_path, 1, My, Image.BILINEAR, "stretched_y.png")

def compress_image_x(image_path, Nx):
    """Сжатие изображения только по горизонтали"""
    resample_image(image_path, 1/Nx, 1, Image.LANCZOS, "compressed_x.png")

def compress_image_y(image_path, Ny):
    """Сжатие изображения только по вертикали"""
    resample_image(image_path, 1, 1/Ny, Image.LANCZOS, "compressed_y.png")

def resample_two_pass(image_path, Mx, My, Nx, Ny):
    """Передискретизация через растяжение, затем сжатие"""
    temp_path_x = "stretched_x.png"
    temp_path_y = "stretched_y.png"
    
    stretch_image_x(image_path, Mx)
    stretch_image_y(temp_path_x, My)
    compress_image_x(temp_path_y, Nx)
    compress_image_y(temp_path_y, 5)

def resample_one_pass(image_path, Kx, Ky):
    """Передискретизация за один проход"""
    resample_image(image_path, Kx, Ky, Image.BICUBIC, "resampled_one_pass.png")

# Запуск функций с тестовым изображением
image_path = "image.png"  # Укажите путь к вашему изображению

# Примеры передискретизации
stretch_image_x(image_path, 2)  # Растяжение по X в 2 раза
stretch_image_y(image_path, 3)  # Растяжение по Y в 3 раза
compress_image_x(image_path, 2)  # Сжатие по X в 2 раза
compress_image_y(image_path, 3)  # Сжатие по Y в 3 раза
resample_two_pass(image_path, 2, 3, 2, 3)  # Передискретизация через два прохода
resample_one_pass(image_path, 2, 3)  # Передискретизация за один проход

def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(image_path).convert('RGB')
    return np.array(img_src)


def two_step_resampling(img: np.array, numerator: int, denominator: int) -> np.array:
    tmp = one_step_resampling(img, numerator, lambda a, b: a * b, lambda a, b: int(round(a / b)))
    return one_step_resampling(tmp, denominator, lambda a, b: int(round(a / b)), lambda a, b: a * b)


def one_step_resampling(img: np.array, factor: float, f1, f2):
    dimensions = img.shape[0:2]
    new_dimensions = tuple(f1(dimension, factor) for dimension in dimensions)
    new_shape = (*new_dimensions, img.shape[2])
    new_img = np.empty(new_shape)

    for x in range(new_dimensions[0]):
        for y in range(new_dimensions[1]):
            new_img[x, y] = img[ min(f2(x, factor), dimensions[0] - 1), min(f2(y, factor), dimensions[1] - 1)]
    return new_img

def execute(img, f1, f2, number_type=int):
    data_type = np.uint8
    color_model = 'RGB'

    factor = 7
    result = Image.fromarray(one_step_resampling(
        img, factor, f1, f2).astype(data_type), color_model)

    return result
#image_path = "image.png"
#result = execute(image_to_np_array(image_path), lambda a, b: a * b, lambda a, b: int(round(a / b)))
#result.save("res.png")
    

#split_rgb(image_path)
#rgb_to_hsi(image_path)
#invert_intensity(image_path)
"""resample_image(image_path, 5, 2, Image.BILINEAR, "stretched.png")

compress_image(image_path, 5, 1)  # Сжатие по X в 5 раз
resample_two_pass(image_path, 2, 3, 2, 3)  # Передискретизация через два прохода
resample_one_pass(image_path, 2, 3)  # Передискретизация за один проход

"""