import cv2
import os
import matplotlib.pyplot as plt
from factory import *


# =========================
# ЗАГРУЗКА
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "test.jpg")

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Исходное изображение")
plt.axis("off")
plt.show()


# =========================
# КАНАЛЫ
# =========================

R, G, B, w, h = get_rgb_channels(img_rgb)
show_channels(R, G, B, "Исходные")


# =========================
# БИНАРИЗАЦИЯ
# =========================

binary, Rb, Gb, Bb = binarize_rgb_image(img_rgb, 128)
show_channels(Rb * 255, Gb * 255, Bb * 255, "Бинаризация")


# =========================
# ЯРКОСТНЫЕ СРЕЗЫ
# =========================

for t in [1, 2, 3]:
    Rs, Gs, Bs = apply_slicing_to_rgb(img_rgb, 120, 200, t)
    show_channels(Rs, Gs, Bs, f"Срез тип {t}")


# =========================
# ЛИНЕЙНЫЙ КОНТРАСТ
# =========================

y_min = 0
y_max = 160
R_lc, G_lc, B_lc = linear_contrast_rgb_channels(img_rgb, y_min, y_max)
show_channels(R_lc, G_lc, B_lc)

plt.imshow(np.stack([R_lc, G_lc, B_lc], axis=2))
plt.title(f'Линейное контрастирование [{y_min}, {y_max}] RGB')
plt.axis('off')
plt.show()
# =========================
# ПИЛООБРАЗНОЕ
# =========================

for mode in [1, 2, 3, 4]:
    Rsc, Gsc, Bsc, rgb_sc = saw_contrast_color(img_rgb, mode)
    show_channels(Rsc, Gsc, Bsc, f"Пилообразное тип {mode} по каналам")
    plt.imshow(rgb_sc)
    plt.title(f"Пилообразное тип {mode} RGB")
    plt.axis("off")
    plt.show()


# =========================
# СОЛЯРИЗАЦИЯ
# =========================

R_solar, G_solar, B_solar, img_solar = solarize_image(img_rgb)
show_channels(R_solar, G_solar, B_solar)
plt.imshow(img_solar)
plt.title('Соляризация RGB')
plt.axis('off')
plt.show()


# =========================
# ГАММА (GRAYSCALE)
# =========================

gray_gamma = gamma_correction_gray(img_rgb, gamma=3)
plt.imshow(gray_gamma, cmap="gray")
plt.title("Гамма-коррекция")
plt.axis("off")
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
plt.imshow(gray_world)
plt.title("Серый мир")
plt.axis("off")
plt.show()


# =========================
# ЛОГ-КОРРЕКЦИЯ
# =========================

log_img = log_correction(img_rgb)
plt.imshow(log_img)
plt.title("Логарифмическая коррекция")
plt.axis("off")
plt.show()

