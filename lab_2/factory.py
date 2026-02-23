import numpy as np
import matplotlib.pyplot as plt
import math


# =========================
# БАЗОВЫЕ ОПЕРАЦИИ
# =========================

def get_rgb_channels(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    height, width = img_rgb.shape[:2]
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
    binary = (img_rgb >= threshold).astype(np.uint8)
    R = binary[:, :, 0]
    G = binary[:, :, 1]
    B = binary[:, :, 2]
    return binary, R, G, B


# =========================
# ЯРКОСТНЫЕ СРЕЗЫ
# =========================

def intensity_slicing(channel, low, high, slice_type=1):
    if slice_type == 1:
        result = np.where((channel >= low) & (channel <= high), 255, 0)

    elif slice_type == 2:
        result = np.where((channel >= low) & (channel <= high), 255, channel)

    elif slice_type == 3:
        result = np.where(channel >= low, 255, channel)
    else:
        return []
    return result.astype(np.uint8)


def apply_slicing_to_rgb(img_rgb, low, high, slice_type=1):
    R = intensity_slicing(img_rgb[:, :, 0], low, high, slice_type)
    G = intensity_slicing(img_rgb[:, :, 1], low, high, slice_type)
    B = intensity_slicing(img_rgb[:, :, 2], low, high, slice_type)
    return R, G, B


# =========================
# ЛИНЕЙНОЕ КОНТРАСТИРОВАНИЕ
# =========================

def linear_contrast_rgb_channels(image, y_min=0, y_max=255):
    def stretch(channel):
        x_min = np.min(channel)
        x_max = np.max(channel)

        if x_max == x_min:
            return channel.astype(np.uint8)

        result = ((channel - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min
        return np.clip(result, y_min, y_max).astype(np.uint8)

    R = stretch(image[:, :, 0])
    G = stretch(image[:, :, 1])
    B = stretch(image[:, :, 2])

    return R, G, B


# =========================
# ПИЛООБРАЗНОЕ КОНТРАСТИРОВАНИЕ
# =========================

def saw_contrast_color(img_rgb, mode: int, N: int = 3):
    """
    Пилообразное контрастирование по каждому каналу отдельно.
    """

    def transform(channel):
        channel = channel.astype(np.float32)
        y = np.zeros_like(channel)

        if mode == 1:
            x1, x2 = 80, 200
            mask = (channel >= x1) & (channel <= x2)
            y[mask] = (channel[mask] - x1) * 255.0 / (x2 - x1)
            y[channel > x2] = 100
            y[channel < x1] = 100

        elif mode == 2:
            x1, x2 = 80, 200
            mask = (channel >= x1) & (channel <= x2)
            y[mask] = (channel[mask] - x1) * 255.0 / (x2 - x1)
            y[channel < x1] = 255
            y[channel > x2] = 255

        elif mode == 3:
            x1, x2 = 50, 180
            mask = (channel >= x1) & (channel <= x2)
            y[mask] = (channel[mask] - x1) * 255.0 / (x2 - x1)

        elif mode == 4:
            period = 256 / N
            y = (channel % period) * (255.0 / period)

        else:
            raise ValueError("mode должен быть 1, 2, 3 или 4")

        return np.clip(y, 0, 255).astype(np.uint8)

    R = transform(img_rgb[:, :, 0])
    G = transform(img_rgb[:, :, 1])
    B = transform(img_rgb[:, :, 2])
    rgb = np.stack([R, G, B], axis=2)
    return R, G, B, rgb


# =========================
# СОЛЯРИЗАЦИЯ
# =========================

def solarize_image(img_rgb):
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)

    xmax_R = np.max(R)
    xmax_G = np.max(G)
    xmax_B = np.max(B)

    k_R = 4 / xmax_R if xmax_R != 0 else 0
    k_G = 4 / xmax_G if xmax_G != 0 else 0
    k_B = 4 / xmax_B if xmax_B != 0 else 0

    R_solarized = k_R * R * (xmax_R - R)
    G_solarized = k_G * G * (xmax_G - G)
    B_solarized = k_B * B * (xmax_B - B)

    R_solarized = np.clip(R_solarized, 0, 255).astype(np.uint8)
    G_solarized = np.clip(G_solarized, 0, 255).astype(np.uint8)
    B_solarized = np.clip(B_solarized, 0, 255).astype(np.uint8)

    rgb_solarized = np.stack([R_solarized, G_solarized, B_solarized], axis=2)
    return R_solarized, G_solarized, B_solarized, rgb_solarized


# =========================
# ГАММА-КОРРЕКЦИЯ
# =========================

def gamma_correction_gray(img_rgb, gamma=3.0):
    H, W, C = img_rgb.shape
    img = img_rgb.astype(float)

    corrected = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            R = img[i, j, 0]
            G = img[i, j, 1]
            B = img[i, j, 2]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            gray = gray / 255.0

            gamma_val = gray ** gamma

            corrected[i, j] = gamma_val * 255.0

            if corrected[i, j] > 255:
                corrected[i, j] = 255
            elif corrected[i, j] < 0:
                corrected[i, j] = 0

    return corrected.astype(np.uint8)


# =========================
# ГИСТОГРАММА
# =========================
def compute_color_histogram(img_rgb):
    hist_R = np.zeros(256, dtype=int)
    hist_G = np.zeros(256, dtype=int)
    hist_B = np.zeros(256, dtype=int)

    height, width = img_rgb.shape[:2]

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
    img = img_rgb.astype(np.float32)
    H, W, C = img.shape

    sum_R = 0.0
    sum_G = 0.0
    sum_B = 0.0
    for i in range(H):
        for j in range(W):
            sum_R += img[i, j, 0]
            sum_G += img[i, j, 1]
            sum_B += img[i, j, 2]

    mean_R = sum_R / (H * W)
    mean_G = sum_G / (H * W)
    mean_B = sum_B / (H * W)

    mean_gray = (mean_R + mean_G + mean_B) / 3.0

    k_R = mean_gray / mean_R
    k_G = mean_gray / mean_G
    k_B = mean_gray / mean_B

    corrected = np.zeros_like(img)
    for i in range(H):
        for j in range(W):
            corrected[i, j, 0] = img[i, j, 0] * k_R
            corrected[i, j, 1] = img[i, j, 1] * k_G
            corrected[i, j, 2] = img[i, j, 2] * k_B

    for i in range(H):
        for j in range(W):
            for c in range(3):
                if corrected[i, j, c] > 255:
                    corrected[i, j, c] = 255
                elif corrected[i, j, c] < 0:
                    corrected[i, j, c] = 0

    return corrected.astype(np.uint8)


# =========================
# ЛОГ-КОРРЕКЦИЯ
# =========================

def log_correction(img_rgb):
    img = img_rgb.astype(np.float32)
    H, W, C = img.shape

    max_val = 0.0
    for i in range(H):
        for j in range(W):
            for c in range(C):
                if img[i, j, c] > max_val:
                    max_val = img[i, j, c]

    c = 255.0 / math.log(1 + max_val)

    log_img = np.zeros_like(img)
    for i in range(H):
        for j in range(W):
            for c_idx in range(C):
                val = img[i, j, c_idx]
                log_val = c * math.log(1 + val)

                if log_val > 255:
                    log_val = 255
                elif log_val < 0:
                    log_val = 0

                log_img[i, j, c_idx] = log_val

    return log_img.astype(np.uint8)
