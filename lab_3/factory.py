import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# ЧТЕНИЕ
# =========================
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Файл не найден")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================
# ОТОБРАЖЕНИЕ КАНАЛОВ
# =========================
def show_channels(img, title="Image"):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)

    fig, axs = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(title)

    axs[0].imshow(img)
    axs[0].set_title("RGB")

    axs[1].imshow(r, cmap="gray")
    axs[1].set_title("R")

    axs[2].imshow(g, cmap="gray")
    axs[2].set_title("G")

    axs[3].imshow(b, cmap="gray")
    axs[3].set_title("B")

    axs[4].imshow(gray, cmap="gray")
    axs[4].set_title("Gray")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# МАСШТАБИРОВАНИЕ
# =========================
def scale_image(img, alpha):
    h, w, c = img.shape
    new_h = int(h * alpha)
    new_w = int(w * alpha)

    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            src_x = int(x / alpha)
            src_y = int(y / alpha)

            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result


# =========================
# ПОВОРОТ ВОКРУГ НАЧАЛА
# =========================
def rotate_origin(img, angle_deg):
    angle = np.radians(angle_deg)
    h, w, c = img.shape

    result = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            src_x = int(x * np.cos(angle) + y * np.sin(angle))
            src_y = int(-x * np.sin(angle) + y * np.cos(angle))

            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result


# =========================
# ПОВОРОТ ВОКРУГ ЦЕНТРА
# =========================
def rotate_center(img, angle_deg):
    angle = np.radians(angle_deg)
    h, w, c = img.shape

    cx = w // 2
    cy = h // 2

    result = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            x_shift = x - cx
            y_shift = y - cy

            src_x = int(x_shift * np.cos(angle) + y_shift * np.sin(angle) + cx)
            src_y = int(-x_shift * np.sin(angle) + y_shift * np.cos(angle) + cy)

            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = img[src_y, src_x]

    return result


# =========================
# ОТРАЖЕНИЯ
# =========================
def flip_horizontal(img):
    h, w, c = img.shape
    result = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            result[h - y - 1, x] = img[y, x]

    return result


def flip_vertical(img):
    h, w, c = img.shape
    result = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            result[y, w - x - 1] = img[y, x]

    return result


# =========================
# СКОС (shear)
# =========================
def shear_image(img, sx=1.0, sy=1.0, shx=0.0, shy=0.0):
    """
    Скос и масштабирование изображения по формулам:
    x' = sx * x + shy * y
    y' = shx * x + sy * y
    """
    h, w, c = img.shape

    new_w = int(abs(sx)*w + abs(shy)*h)
    new_h = int(abs(sy)*h + abs(shx)*w)

    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    det = sx * sy - shx * shy
    if det == 0:
        raise ValueError("Невозможно выполнить преобразование: det=0")

    for y_prime in range(new_h):
        for x_prime in range(new_w):

            src_x = (sy * x_prime - shy * y_prime) / det
            src_y = (-shx * x_prime + sx * y_prime) / det

            src_x_int = int(round(src_x))
            src_y_int = int(round(src_y))

            if 0 <= src_x_int < w and 0 <= src_y_int < h:
                result[y_prime, x_prime] = img[src_y_int, src_x_int]

    return result
