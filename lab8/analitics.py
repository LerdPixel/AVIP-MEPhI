import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка изображения (в BGR-формате по умолчанию в OpenCV)
img_bgr = cv2.imread('your_image.jpg')  # замените на путь к вашему изображению
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Преобразование в HLS (не путать с HSL — в OpenCV используется HLS)
img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

# 3. Извлечение канала яркости L
l_channel = img_hls[:, :, 1]

# 4. Отображение
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Оригинал (RGB)")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Яркостной канал (L)")
plt.imshow(l_channel, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Гистограмма яркости (L)")
plt.hist(l_channel.ravel(), bins=256, range=(0, 255), color='black')
plt.grid(True)

plt.tight_layout()
plt.show()
