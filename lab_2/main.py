import cv2
import os
import matplotlib.pyplot as plt
from factory import *

# Чтение изображения
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "test.jpg")

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# Отображаем изображение по каналам
red, green, blue, width, height = get_rgb_channels(img_rgb)
original_red = parse_grayscale(red)
original_green = parse_grayscale(green)
original_blue = parse_grayscale(blue)
show_channels(original_red, original_green, original_blue)

# Бинаризация
threshold = 128
binary_image, red_bin, green_bin, blue_bin = binarize_rgb_image(img_rgb, width, height, threshold)
show_binarization_results(img_rgb, binary_image, threshold)
show_channels(parse_grayscale_bin(red_bin), parse_grayscale_bin(green_bin), parse_grayscale_bin(blue_bin))

# Яркостные срезы
slice_type = 1
low = 127
high = 200
red1, green1, blue1 = apply_slicing_to_rgb(img_rgb, low, high, height, width, slice_type)

show_slicing_results(original_red, original_green, original_blue,
                     parse_grayscale(red1), parse_grayscale(green1), parse_grayscale(blue1), slice_type)
slice_type = 2
red2, green2, blue2 = apply_slicing_to_rgb(img_rgb, low, high, height, width, slice_type)

show_slicing_results(original_red, original_green, original_blue,
                     parse_grayscale(red2), parse_grayscale(green2), parse_grayscale(blue2), slice_type)

slice_type = 3
red3, green3, blue3 = apply_slicing_to_rgb(img_rgb, low, high, height, width, slice_type)

show_slicing_results(original_red, original_green, original_blue,
                     parse_grayscale(red3), parse_grayscale(green3), parse_grayscale(blue3), slice_type)

# Линейное контрастирование
y_max = 160
red_stretched, green_stretched, blue_stretched = linear_contrast_rgb_channels(img_rgb)

show_linear_contrast_results(original_red, original_green, original_blue,
                             parse_grayscale(
                                 red_stretched), parse_grayscale(green_stretched), parse_grayscale(blue_stretched),
                             0, y_max)

# Пилообразное контрастирование
show_image_saw_contrast(img_rgb, 1)
show_image_saw_contrast(img_rgb, 2)
show_image_saw_contrast(img_rgb, 3)
show_image_saw_contrast(img_rgb, 4)

# Соляризация цветного изображения
red_solarized, green_solarize, blue_solarized, rgb_solarize = solarize_image(img_rgb)

show_solarization_results(parse_grayscale(red_solarized), parse_grayscale(green_solarize),
                          parse_grayscale(blue_solarized), rgb_solarize)

# Гамма коррекция
gamma = 2
r, g, b, rgb_gamma = gamma_correction(img_rgb, gamma)
show_gamma_correction_results(parse_grayscale(r), parse_grayscale(g), parse_grayscale(b), rgb_gamma, gamma)

# Гистрограмма
hist_r, hist_g, hist_b = compute_color_histogram(img_rgb)
show_color_histogram(hist_r, hist_g, hist_b)

# Коррекция серый мир
img_gray = gray_correction(img_rgb)
show_image_gray_correction(img_rgb, img_gray)

# Логарифм
img_log = log_correction(img_rgb)
show_image_log_correction(img_rgb, img_log)
