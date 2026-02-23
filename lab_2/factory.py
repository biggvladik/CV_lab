import numpy as np
import matplotlib.pyplot as plt
import math

# =========================
# БАЗОВЫЕ ОПЕРАЦИИ
# =========================

def get_rgb_channels(img_rgb):
    height, width = img_rgb.shape[:2]
    R = np.zeros((height, width), dtype=np.uint8)
    G = np.zeros((height, width), dtype=np.uint8)
    B = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            R[y, x] = img_rgb[y, x, 0]
            G[y, x] = img_rgb[y, x, 1]
            B[y, x] = img_rgb[y, x, 2]
    return R, G, B, width, height


def show_channels(R, G, B, title_prefix=""):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(R, cmap="gray")
    plt.title(f"{title_prefix} R")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(G, cmap="gray")
    plt.title(f"{title_prefix} G")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(B, cmap="gray")
    plt.title(f"{title_prefix} B")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# БИНАРИЗАЦИЯ
# =========================

def binarize_rgb_image(img_rgb, threshold=128):
    height, width = img_rgb.shape[:2]
    binary = np.zeros((height, width, 3), dtype=np.uint8)
    R = np.zeros((height, width), dtype=np.uint8)
    G = np.zeros((height, width), dtype=np.uint8)
    B = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                val = img_rgb[y, x, c]
                if val >= threshold:
                    binary[y, x, c] = 255
                else:
                    binary[y, x, c] = 0
            R[y, x] = binary[y, x, 0]
            G[y, x] = binary[y, x, 1]
            B[y, x] = binary[y, x, 2]
    return binary, R, G, B


# =========================
# ЯРКОСТНЫЕ СРЕЗЫ
# =========================

def intensity_slicing(channel, low, high, slice_type=1):
    height, width = channel.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            val = channel[y, x]
            if slice_type == 1:
                result[y, x] = 255 if low <= val <= high else 0
            elif slice_type == 2:
                result[y, x] = 255 if low <= val <= high else val
            elif slice_type == 3:
                result[y, x] = 255 if val >= low else val
            else:
                result[y, x] = 0
    return result


def apply_slicing_to_rgb(img_rgb, low, high, slice_type=1):
    R = intensity_slicing(img_rgb[:, :, 0], low, high, slice_type)
    G = intensity_slicing(img_rgb[:, :, 1], low, high, slice_type)
    B = intensity_slicing(img_rgb[:, :, 2], low, high, slice_type)
    return R, G, B


# =========================
# ЛИНЕЙНОЕ КОНТРАСТИРОВАНИЕ
# =========================

def linear_contrast_rgb_channels(img_rgb, y_min=0, y_max=255):
    height, width = img_rgb.shape[:2]

    def stretch(channel):
        min_val = 255
        max_val = 0
        for y in range(height):
            for x in range(width):
                val = channel[y, x]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

        stretched = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                val = channel[y, x]
                if max_val == min_val:
                    stretched[y, x] = val
                else:
                    tmp = int((val - min_val) / (max_val - min_val) * (y_max - y_min) + y_min)
                    stretched[y, x] = min(max(tmp, 0), 255)
        return stretched

    R = stretch(img_rgb[:, :, 0])
    G = stretch(img_rgb[:, :, 1])
    B = stretch(img_rgb[:, :, 2])
    return R, G, B


# =========================
# ПИЛООБРАЗНОЕ КОНТРАСТИРОВАНИЕ
# =========================

def saw_contrast_color(img_rgb, mode, N=3):
    height, width = img_rgb.shape[:2]

    def transform(channel):
        result = np.zeros((height, width), dtype=np.uint8)
        if mode == 1:
            x1, x2 = 80, 200
            for y in range(height):
                for x in range(width):
                    val = int(channel[y, x])
                    if val < x1 or val > x2:
                        result[y, x] = 100
                    else:
                        temp = (val - x1) * 255 / (x2 - x1)
                        result[y, x] = np.clip(int(temp), 0, 255)
        elif mode == 2:
            x1, x2 = 80, 200
            for y in range(height):
                for x in range(width):
                    val = int(channel[y, x])
                    if val < x1 or val > x2:
                        result[y, x] = 255
                    else:
                        temp = (val - x1) * 255 / (x2 - x1)
                        result[y, x] = np.clip(int(temp), 0, 255)
        elif mode == 3:
            x1, x2 = 50, 180
            for y in range(height):
                for x in range(width):
                    val = int(channel[y, x])
                    if x1 <= val <= x2:
                        temp = (val - x1) * 255 / (x2 - x1)
                        result[y, x] = np.clip(int(temp), 0, 255)
                    else:
                        result[y, x] = 0
        elif mode == 4:
            period = 256 / N
            for y in range(height):
                for x in range(width):
                    val = int(channel[y, x])
                    temp = (val % period) * 255 / period
                    result[y, x] = np.clip(int(temp), 0, 255)
        else:
            raise ValueError("mode должен быть 1,2,3 или 4")
        return result

    R = transform(img_rgb[:, :, 0])
    G = transform(img_rgb[:, :, 1])
    B = transform(img_rgb[:, :, 2])
    rgb = np.stack([R, G, B], axis=2)
    return R, G, B, rgb


# =========================
# СОЛЯРИЗАЦИЯ
# =========================

def solarize_image(img_rgb):
    height, width = img_rgb.shape[:2]
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)

    xmax_R = 0
    xmax_G = 0
    xmax_B = 0
    for y in range(height):
        for x in range(width):
            if R[y, x] > xmax_R: xmax_R = R[y, x]
            if G[y, x] > xmax_G: xmax_G = G[y, x]
            if B[y, x] > xmax_B: xmax_B = B[y, x]

    k_R = 4 / xmax_R if xmax_R != 0 else 0
    k_G = 4 / xmax_G if xmax_G != 0 else 0
    k_B = 4 / xmax_B if xmax_B != 0 else 0

    R_solarized = np.zeros((height, width), dtype=np.uint8)
    G_solarized = np.zeros((height, width), dtype=np.uint8)
    B_solarized = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            R_solarized[y, x] = int(k_R * R[y, x] * (xmax_R - R[y, x]))
            G_solarized[y, x] = int(k_G * G[y, x] * (xmax_G - G[y, x]))
            B_solarized[y, x] = int(k_B * B[y, x] * (xmax_B - B[y, x]))

    rgb_solarized = np.stack([R_solarized, G_solarized, B_solarized], axis=2)
    return R_solarized, G_solarized, B_solarized, rgb_solarized


# =========================
# ГАММА-КОРРЕКЦИЯ
# =========================

def gamma_correction_gray(img_rgb, gamma=3.0):
    height, width = img_rgb.shape[:2]
    corrected = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            R = img_rgb[y, x, 0]
            G = img_rgb[y, x, 1]
            B = img_rgb[y, x, 2]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            gray /= 255.0
            val = gray ** gamma * 255
            if val > 255: val = 255
            if val < 0: val = 0
            corrected[y, x] = int(val)
    return corrected


# =========================
# ГИСТОГРАММА
# =========================

def compute_color_histogram(img_rgb):
    height, width = img_rgb.shape[:2]
    hist_R = np.zeros(256, dtype=int)
    hist_G = np.zeros(256, dtype=int)
    hist_B = np.zeros(256, dtype=int)

    for y in range(height):
        for x in range(width):
            r = img_rgb[y, x, 0]
            g = img_rgb[y, x, 1]
            b = img_rgb[y, x, 2]
            hist_R[r] += 1
            hist_G[g] += 1
            hist_B[b] += 1
    return hist_R, hist_G, hist_B


def show_color_histogram(hist_R, hist_G, hist_B):
    plt.figure(figsize=(8, 4))
    plt.plot(hist_R, color='red', label='R')
    plt.plot(hist_G, color='green', label='G')
    plt.plot(hist_B, color='blue', label='B')
    plt.legend()
    plt.grid(True)
    plt.title("Гистограмма RGB")
    plt.show()


# =========================
# СЕРЫЙ МИР
# =========================

def gray_world_correction(img_rgb):
    height, width = img_rgb.shape[:2]
    sum_R = sum_G = sum_B = 0.0
    for y in range(height):
        for x in range(width):
            sum_R += img_rgb[y, x, 0]
            sum_G += img_rgb[y, x, 1]
            sum_B += img_rgb[y, x, 2]

    mean_R = sum_R / (height * width)
    mean_G = sum_G / (height * width)
    mean_B = sum_B / (height * width)
    mean_gray = (mean_R + mean_G + mean_B) / 3.0

    corrected = np.zeros_like(img_rgb, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            val_R = int(img_rgb[y, x, 0] * mean_gray / mean_R)
            val_G = int(img_rgb[y, x, 1] * mean_gray / mean_G)
            val_B = int(img_rgb[y, x, 2] * mean_gray / mean_B)
            corrected[y, x, 0] = min(max(val_R, 0), 255)
            corrected[y, x, 1] = min(max(val_G, 0), 255)
            corrected[y, x, 2] = min(max(val_B, 0), 255)
    return corrected


# =========================
# ЛОГ-КОРРЕКЦИЯ
# =========================

def log_correction(img_rgb):
    img_rgb_float = img_rgb.astype(np.float32)

    max_val = np.max(img_rgb_float)

    if max_val <= 0:
        max_val = 1

    c_factor = 255 / math.log(1 + max_val)

    log_img = c_factor * np.log(1 + img_rgb_float)

    log_img = np.clip(log_img, 0, 255).astype(np.uint8)

    return log_img
