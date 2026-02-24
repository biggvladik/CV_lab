import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from factory import *

# =========================
# ЗАГРУЗКА
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "test.jpg")

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# БИНАРИЗАЦИЯ
# =========================

binary, Rb, Gb, Bb = binarize_rgb_image(img_rgb, 128)
rgb_binary = np.stack([Rb, Gb, Bb], axis=2).astype(np.uint8)

show_comparison(img_rgb, rgb_binary, "Бинаризация")

# =========================
# ЯРКОСТНЫЕ СРЕЗЫ (RGB)
# =========================

for t in [1, 2, 3]:
    Rs, Gs, Bs = apply_slicing_to_rgb(img_rgb, 120, 200, t)
    rgb_slice = np.stack([Rs, Gs, Bs], axis=2).astype(np.uint8)

    show_comparison(img_rgb, rgb_slice, f"Срез тип {t}")

# =========================
# ЛИНЕЙНЫЙ КОНТРАСТ
# =========================

y_min = 0
y_max = 160

R_lc, G_lc, B_lc = linear_contrast_rgb_channels(img_rgb, y_min, y_max)
rgb_lc = np.stack([R_lc, G_lc, B_lc], axis=2).astype(np.uint8)

show_comparison(img_rgb, rgb_lc, f"Линейный контраст [{y_min},{y_max}]")

# =========================
# ПИЛООБРАЗНОЕ
# =========================

for mode in [1, 2, 3, 4]:
    Rsc, Gsc, Bsc, rgb_sc = saw_contrast_color(img_rgb, mode)
    show_comparison(img_rgb, rgb_sc, f"Пилообразное тип {mode}")

# =========================
# СОЛЯРИЗАЦИЯ
# =========================

R_solar, G_solar, B_solar, img_solar = solarize_image(img_rgb)
show_comparison(img_rgb, img_solar, "Соляризация")

# =========================
# ГАММА (GRAYSCALE)
# =========================

gray_gamma = gamma_correction_gray(img_rgb, gamma=3)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Оригинал RGB")
axes[0].axis("off")

axes[1].imshow(gray_gamma, cmap="gray")
axes[1].set_title("Гамма-коррекция (Gray)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# =========================
# ГИСТОГРАММА
# =========================

hist_R, hist_G, hist_B = compute_color_histogram(img_rgb)
show_color_histogram(hist_R, hist_G, hist_B)

# =========================
# СЕРЫЙ МИР
# =========================

gray_world = gray_world_correction(img_rgb)
show_comparison(img_rgb, gray_world, "Серый мир")

# =========================
# ЛОГ-КОРРЕКЦИЯ
# =========================

log_img = log_correction(img_rgb)
show_comparison(img_rgb, log_img, "Логарифмическая коррекция")
