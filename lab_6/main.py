import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from factory import *

# =========================
# Загрузка изображения
# =========================
def load_grayscale(path):
    img = Image.open(path).convert('L')
    return np.array(img)


# =========================
# Оцу
# =========================
def otsu_threshold(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    total = img.size

    sum_total = np.sum(np.arange(256) * hist)

    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]

        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold


def binarize(img):
    t = otsu_threshold(img)
    return np.where(img > t, 255, 0).astype(np.uint8)


# =========================
# УНИВЕРСАЛЬНЫЙ ВЫВОД
# =========================
def show_operation(title, original, results):
    plt.figure(figsize=(14, 6))

    # оригинал
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    # результаты
    for i, (name, img) in enumerate(results.items(), start=2):
        plt.subplot(2, 3, i)
        plt.title(name)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.suptitle(title)
    plt.show()


# =========================
# MAIN
# =========================
def main():
    img = load_grayscale("image_1.jpg")
    binary = binarize(img)

    kernels = {
        "Прямоугольник": rectangle_kernel(10, 10),
        "Крест": cross_kernel(10),
        "Диск": disk_kernel(10),
        "Кольцо": ring_kernel(10),
        "Ромб": diamond_kernel(10)
    }

    # =========================
    # ДИЛАТАЦИЯ
    # =========================
    results = {name: dilate(binary, k) for name, k in kernels.items()}
    show_operation("Dilation", binary, results)

    # =========================
    # ЭРОЗИЯ
    # =========================
    results = {name: erode(binary, k) for name, k in kernels.items()}
    show_operation("Erosion", binary, results)

    # =========================
    # CLOSING
    # =========================
    results = {name: closing(binary, k) for name, k in kernels.items()}
    show_operation("Closing", binary, results)

    # =========================
    # OPENING
    # =========================
    results = {name: opening(binary, k) for name, k in kernels.items()}
    show_operation("Opening", binary, results)

    # =========================
    # РЕКОНСТРУКЦИЯ
    # =========================
    results = {}
    for name, k in kernels.items():
        marker = erode(binary, k)
        results[name] = reconstruction(marker, binary, k)

    show_operation("Reconstruction", binary, results)

    # =========================
    # ГРАДИЕНТЫ
    # =========================
    results_ext = {name: gradient_external(binary, k) for name, k in kernels.items()}
    show_operation("Gradient External", binary, results_ext)

    results_int = {name: gradient_internal(binary, k) for name, k in kernels.items()}
    show_operation("Gradient Internal", binary, results_int)

    results_full = {name: gradient_full(binary, k) for name, k in kernels.items()}
    show_operation("Gradient Full", binary, results_full)


if __name__ == "__main__":
    main()