import numpy as np
from PIL import Image


# =========================
# IMAGE IO
# =========================
def read_image(path):
    return np.array(Image.open(path).convert("RGB"))


def to_grayscale(img):
    return (0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2]).astype(np.uint8)


# =========================
# PADDING
# =========================
def pad_image(img, pad, mode):

    h, w, c = img.shape
    new_h, new_w = h + 2 * pad, w + 2 * pad

    # Создаём пустой массив
    padded = np.zeros((new_h, new_w, c), dtype=img.dtype)

    # вставляем исходное изображение в центр
    padded[pad:pad + h, pad:pad + w, :] = img

    if mode == "clip":  # просто оставляем нули на границах
        return padded

    elif mode == "copy":  # повторяем крайние пиксели
        # верх
        padded[:pad, pad:pad + w, :] = img[0:1, :, :]
        # низ
        padded[pad + h:, pad:pad + w, :] = img[-1:, :, :]
        # левый и правый
        padded[:, :pad, :] = padded[:, pad:pad + 1, :]
        padded[:, pad + w:, :] = padded[:, pad + w - 1:pad + w, :]
        return padded

    elif mode == "reflect":  # зеркальное отражение
        # вертикальное отражение
        for i in range(pad):
            padded[pad - i - 1, pad:pad + w, :] = img[i, :, :]  # верх
            padded[pad + h + i, pad:pad + w, :] = img[h - 2 - i, :, :]  # низ
        # горизонтальное отражение
        for j in range(pad):
            padded[:, pad - j - 1, :] = padded[:, pad + j, :]  # левый
            padded[:, pad + w + j, :] = padded[:, pad + w - 2 - j, :]  # правый
        return padded

    elif mode == "wrap":  # циклическое оборачивание
        # вертикальные края
        padded[:pad, pad:pad + w, :] = img[-pad:, :, :]  # верх
        padded[pad + h:, pad:pad + w, :] = img[:pad, :, :]  # низ
        # горизонтальные края
        padded[:, :pad, :] = padded[:, w:w + pad, :]  # левый
        padded[:, pad + w:, :] = padded[:, pad:pad + pad, :]  # правый
        return padded

    else:
        raise ValueError(f"Unknown padding mode: {mode}")


# =========================
# ORDER FILTER
# =========================
def order_filter(img, ksize=3, mode="reflect", ftype="median"):
    pad = ksize // 2
    h, w, c = img.shape

    # Добавляем паддинг
    padded = pad_image(img, pad, mode)

    # Результат
    result = np.zeros_like(img)

    # Проходим по всем пикселям исходного изображения
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                # Берём окно ksize x ksize вокруг текущего пикселя
                window = padded[i:i + ksize, j:j + ksize, ch]

                # Вычисляем нужную статистику
                if ftype == "median":
                    result[i, j, ch] = np.median(window)
                elif ftype == "min":
                    result[i, j, ch] = np.min(window)
                elif ftype == "max":
                    result[i, j, ch] = np.max(window)

    return result.astype(np.uint8)


# =========================
# CONVOLUTION
# =========================
def convolve(img, kernel, mode="reflect"):
    kernel = np.array(kernel)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    h, w, c = img.shape
    padded = pad_image(img, max(pad_h, pad_w), mode)
    result = np.zeros((h, w, c), dtype=float)

    # Проходим по всем каналам
    for ch in range(c):
        # Проходим по всем пикселям исходного изображения
        for i in range(h):
            for j in range(w):
                acc = 0.0
                # Проходим по ядру
                for ki in range(kh):
                    for kj in range(kw):
                        # Берём пиксель из padded с соответствующим сдвигом
                        pi = i + ki
                        pj = j + kj
                        acc += kernel[ki, kj] * padded[pi, pj, ch]
                # Сохраняем результат
                result[i, j, ch] = acc

    # Ограничиваем диапазон [0, 255] и приводим к uint8
    return np.clip(result, 0, 255).astype(np.uint8)


# =========================
# GAUSSIAN
# =========================
def gaussian3():
    return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


def gaussian5():
    return np.array([
        [2, 7, 12, 7, 2],
        [7, 31, 52, 31, 7],
        [12, 52, 127, 52, 12],
        [7, 31, 52, 31, 7],
        [2, 7, 12, 7, 2]
    ]) / 571


# =========================
# GRADIENTS
# =========================
def gradient(img, kx, ky):
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return np.clip(np.sqrt(gx.astype(float) ** 2 + gy.astype(float) ** 2), 0, 255).astype(np.uint8)


# Prewitt
def prewitt_horizontal(img):
    kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    return convolve(img, kx)


def prewitt_vertical(img):
    ky = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    return convolve(img, ky)


def prewitt_combined(img):
    return gradient(img,
                    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                    [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


# Sobel
def sobel_horizontal(img):
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    return convolve(img, kx)


def sobel_vertical(img):
    ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    return convolve(img, ky)


def sobel_combined(img):
    return gradient(img,
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# Scharr
def scharr_horizontal(img):
    kx = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
    return convolve(img, kx)


def scharr_vertical(img):
    ky = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
    return convolve(img, ky)


def scharr_combined(img):
    return gradient(img,
                    [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                    [[3, 10, 3], [0, 0, 0], [-3, -10, -3]])


# =========================
# KIRSCH
# =========================
def kirsch(img):
    kernels = [
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
        [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]
    ]
    responses = [convolve(img, k) for k in kernels]
    return np.max(responses, axis=0)


# =========================
# LAPLACE
# =========================
def laplace3():
    return [[0, 1, 0], [1, -4, 1], [0, 1, 0]]


def laplace5():
    return [
        [-1, -3, -4, -3, -1],
        [-3, 0, 6, 0, -3],
        [-4, 6, 20, 6, -4],
        [-3, 0, 6, 0, -3],
        [-1, -3, -4, -3, -1]
    ]


# =========================
# HIGH PASS
# =========================
def highpass3():
    return [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]


def highpass5():
    return [
        [-1] * 5,
        [-1] * 5,
        [-1, -1, 24, -1, -1],
        [-1] * 5,
        [-1] * 5
    ]


# =========================
# SHARPEN
# =========================
def unsharp(img, k=1.0):
    blur = convolve(img, gaussian3())
    return np.clip(img + k * (img - blur), 0, 255).astype(np.uint8)


# =========================
# EMBOSS
# =========================
def emboss(img):
    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    result = convolve(img, kernel)
    return np.clip(result + 128, 0, 255).astype(np.uint8)


# =========================
# NOISE
# =========================
def salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    h, w, c = img.shape
    num_pixels = int(amount * h * w)
    # белые точки
    coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    noisy[coords] = 255
    # черные точки
    coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    noisy[coords] = 0
    return noisy
