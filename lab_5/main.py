import matplotlib.pyplot as plt
from factory import *

img = read_image("image_1.jpg")

# =========================
# Каналы и серое изображение
# =========================
plt.figure(figsize=(10, 4))
for i, (data, title) in enumerate([
    (img[:, :, 0], "R"),
    (img[:, :, 1], "G"),
    (img[:, :, 2], "B"),
    (to_grayscale(img), "Gray")
]):
    plt.subplot(1, 4, i + 1)
    plt.imshow(data, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.show()

# =========================
# Применение всех фильтров
# =========================
results = {
    "Median": order_filter(img, 10, "reflect", "median"),
    "Min": order_filter(img, 10, "reflect", "min"),
    "Max": order_filter(img, 10, "reflect", "max"),
    "Gauss3": convolve(img, gaussian3()),
    "Gauss5": convolve(img, gaussian5()),
    "Roberts": gradient(img, [[1, 0], [0, -1]], [[0, 1], [-1, 0]]),
    "Kirsch": kirsch(img),
    "Laplace3": convolve(img, laplace3()),
    "Laplace5": convolve(img, laplace5()),
    "HP3": convolve(img, highpass3()),
    "HP5": convolve(img, highpass5()),
    "Sharp": unsharp(img, 1.5),
    "Emboss": emboss(img)
}

# =========================
# Вывод результатов всех фильтров
# =========================
plt.figure(figsize=(15, 10))
for i, (name, im) in enumerate(results.items()):
    plt.subplot(4, 4, i + 1)
    plt.imshow(im)
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.show()

# =========================
# Отдельные градиенты Prewitt, Sobel, Scharr
# =========================
gradients = {
    "Prewitt Horizontal": prewitt_horizontal(img),
    "Prewitt Vertical": prewitt_vertical(img),
    "Prewitt Combined": prewitt_combined(img),
    "Sobel Horizontal": sobel_horizontal(img),
    "Sobel Vertical": sobel_vertical(img),
    "Sobel Combined": sobel_combined(img),
    "Scharr Horizontal": scharr_horizontal(img),
    "Scharr Vertical": scharr_vertical(img),
    "Scharr Combined": scharr_combined(img)
}

plt.figure(figsize=(18, 8))
for i, (name, im) in enumerate(gradients.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(im)
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.show()

# =========================
# Фильтрация шума "соль и перец"
# =========================
noisy = salt_pepper_noise(img, 0.05)
median = order_filter(noisy, 10, "reflect", "median")
minf = order_filter(noisy, 10, "reflect", "min")
maxf = order_filter(noisy, 10, "reflect", "max")

plt.figure(figsize=(12,4))
titles = ["Noisy", "Median", "Min", "Max"]
images = [noisy, median, minf, maxf]

for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Сравнение фильтров на шуме 'соль и перец'")
plt.show()