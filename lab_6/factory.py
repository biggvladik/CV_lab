import numpy as np


# =========================
# Примитивы
# =========================

def rectangle_kernel(height, width):
    return np.ones((height, width), dtype=np.uint8)


def cross_kernel(size):
    k = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    k[mid, :] = 1
    k[:, mid] = 1
    return k


def disk_kernel(size):
    k = np.zeros((size, size), dtype=np.uint8)
    mid = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            if (i - mid + 0.5) ** 2 + (j - mid + 0.5) ** 2 <= mid ** 2:
                k[i, j] = 1
    return k


def ring_kernel(size):
    k = np.zeros((size, size), dtype=np.uint8)
    mid = (size - 1) / 2
    outer = mid
    inner = mid - 1
    for i in range(size):
        for j in range(size):
            r2 = (i - mid + 0.5) ** 2 + (j - mid + 0.5) ** 2
            if inner ** 2 <= r2 <= outer ** 2:
                k[i, j] = 1
    return k


def diamond_kernel(size):
    k = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - mid) + abs(j - mid) <= mid:
                k[i, j] = 1
    return k


# =========================
# Вспомогательная функция padding
# =========================
def pad_image(img, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    h, w = img.shape

    # создаём новое изображение с нулями
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded = np.zeros((new_h, new_w), dtype=img.dtype)

    # копируем исходное изображение в центр
    for i in range(h):
        for j in range(w):
            padded[i + pad_h][j + pad_w] = img[i][j]

    return padded


# =========================
# ДИЛАТАЦИЯ
# =========================
def dilate(img, kernel):
    img = (img == 255)
    kernel = (kernel == 1)

    padded = pad_image(img, kernel)
    h, w = img.shape
    kh, kw = kernel.shape

    result = np.zeros((h, w), dtype=bool)

    for i in range(h):
        for j in range(w):

            found = False  # нашли ли белый пиксель

            for ki in range(kh):
                for kj in range(kw):

                    if kernel[ki][kj]:
                        if padded[i + ki][j + kj]:
                            found = True
                            break

                if found:
                    break

            result[i][j] = found

    return (result * 255).astype(np.uint8)


# =========================
# ЭРОЗИЯ
# =========================
def erode(img, kernel):
    img = (img == 255)
    kernel = (kernel == 1)

    padded = pad_image(img, kernel)
    h, w = img.shape
    kh, kw = kernel.shape

    result = np.zeros((h, w), dtype=bool)

    for i in range(h):
        for j in range(w):

            fits = True

            for ki in range(kh):
                for kj in range(kw):

                    if kernel[ki][kj]:
                        if not padded[i + ki][j + kj]:
                            fits = False
                            break

                if not fits:
                    break

            result[i][j] = fits

    return (result * 255).astype(np.uint8)


# =========================
# OPENING
# =========================
def opening(img, kernel):
    return dilate(erode(img, kernel), kernel)


# =========================
# CLOSING
# =========================
def closing(img, kernel):
    return erode(dilate(img, kernel), kernel)


# =========================
# РЕКОНСТРУКЦИЯ
# =========================
def reconstruction(marker, mask, kernel, max_iter=100):
    curr = marker.copy()

    for _ in range(max_iter):
        new = dilate(curr, kernel)
        new = np.minimum(new, mask)

        if np.array_equal(new, curr):
            break

        curr = new

    return curr


# =========================
# ГРАДИЕНТЫ
# =========================
def gradient_external(img, kernel):
    return dilate(img, kernel) - img


def gradient_internal(img, kernel):
    return img - erode(img, kernel)


def gradient_full(img, kernel):
    return dilate(img, kernel) - erode(img, kernel)
