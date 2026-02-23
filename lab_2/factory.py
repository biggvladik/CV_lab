import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def get_rgb_channels(img_rgb):
    width, height = get_size(img_rgb)
    red_channel, green_channel, blue_channel = split_channels(img_rgb, width, height)

    return red_channel, green_channel, blue_channel, width, height


def get_size(image):
    bgr_list = image.tolist()

    height = len(bgr_list)

    width = len(bgr_list[0]) if height > 0 else 0

    return width, height


def split_channels(img_rgb, width, height):
    rgb_list = img_rgb.tolist()

    red_channel = []
    green_channel = []
    blue_channel = []

    for i in range(height):
        red_row = []
        green_row = []
        blue_row = []

        for j in range(width):
            pixel = rgb_list[i][j]
            red_row.append(pixel[0])
            green_row.append(pixel[1])
            blue_row.append(pixel[2])

        red_channel.append(red_row)
        green_channel.append(green_row)
        blue_channel.append(blue_row)

    red_array = np.array(red_channel, dtype=np.uint8)
    green_array = np.array(green_channel, dtype=np.uint8)
    blue_array = np.array(blue_channel, dtype=np.uint8)

    return red_array, green_array, blue_array


def parse_grayscale(channel):
    height = len(channel)
    width = len(channel[0])

    grayscale_img = []

    for i in range(height):
        row = []
        for j in range(width):
            brightness = channel[i][j]

            pixel = [brightness, brightness, brightness]
            row.append(pixel)
        grayscale_img.append(row)

    return np.array(grayscale_img, dtype=np.uint8)


def parse_grayscale_bin(channel):
    height = len(channel)
    width = len(channel[0])

    grayscale_img = []

    for i in range(height):
        row = []
        for j in range(width):
            brightness = channel[i][j] * 255
            pixel = [brightness, brightness, brightness]
            row.append(pixel)
        grayscale_img.append(row)

    return np.array(grayscale_img, dtype=np.uint8)


def show_channels(red, green, blue):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(red)
    plt.title('Красный канал')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green)
    plt.title('Зеленый канал')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue)
    plt.title('Синий канал')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def binarize_rgb_image(img_rgb, width, height, threshold=128):
    count_channel = 3

    binary = np.zeros((height, width, count_channel), dtype=np.uint8)

    red_bin = np.zeros((height, width), dtype=np.uint8)
    green_bin = np.zeros((height, width), dtype=np.uint8)
    blue_bin = np.zeros((height, width), dtype=np.uint8)

    for x in range(height):
        for y in range(width):
            for z in range(count_channel):
                pixel_value = img_rgb[x, y, z]

                if pixel_value >= threshold:
                    binary[x, y, z] = 1

                    if z == 0:
                        red_bin[x, y] = 1
                    elif z == 1:
                        green_bin[x, y] = 1
                    else:
                        blue_bin[x, y] = 1
                else:
                    binary[x, y, z] = 0

                    if z == 0:
                        red_bin[x, y] = 0
                    elif z == 1:
                        green_bin[x, y] = 0
                    else:
                        blue_bin[x, y] = 0

    return binary, red_bin, green_bin, blue_bin


def show_binarization_results(original, binary_img, threshold):
    binary_display = (binary_img * 255).astype(np.uint8)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Оригинал')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_display)
    plt.title(f'Бинаризация (порог = {threshold})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def intensity_slicing(channel, low, high, height, width, slice_type=1):
    """
    1 - бинарный: вне диапазона -> 0, внутри -> 255
    2 - с сохранением фона: внутри -> 255, вне -> исходные
    3 - порог с сохранением фона (как на слайде):
        если pixel >= low -> 255
        иначе -> исходное значение
    """

    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = channel[i][j]

            if slice_type == 1:
                # 1 тип
                if low <= pixel <= high:
                    result[i][j] = 255
                else:
                    result[i][j] = 0

            elif slice_type == 2:
                # 2 тип
                if low <= pixel <= high:
                    result[i][j] = 255
                else:
                    result[i][j] = pixel

            elif slice_type == 3:
                # 3 тип
                if pixel >= low:
                    result[i][j] = 255
                else:
                    result[i][j] = pixel

    return result


def apply_slicing_to_rgb(img_rgb, low, high, height, width, slice_type=1):
    red = img_rgb[:, :, 0]
    green = img_rgb[:, :, 1]
    blue = img_rgb[:, :, 2]

    red_sliced = intensity_slicing(red, low, high, height, width, slice_type)
    green_sliced = intensity_slicing(green, low, high, height, width, slice_type)
    blue_sliced = intensity_slicing(blue, low, high, height, width, slice_type)

    return red_sliced, green_sliced, blue_sliced


def show_slicing_results(red, green, blue,
                         red_sliced, green_sliced, blue_sliced, slice_type):
    plt.figure(figsize=(10, 5))

    # Оригинальные каналы
    plt.subplot(2, 3, 1)
    plt.imshow(red)
    plt.title('Исходный RED')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(green)
    plt.title('Исходный GREEN')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(blue)
    plt.title('Исходный BLUE')
    plt.axis('off')

    # Результаты среза
    plt.subplot(2, 3, 4)
    plt.imshow(red_sliced)
    plt.title(f'RED после среза {slice_type}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(green_sliced)
    plt.title(f'GREEN после среза {slice_type}')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(blue_sliced)
    plt.title(f'BLUE после среза {slice_type}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def linear_contrast_rgb_channels(image, y_min=0, y_max=255):
    # разделяем каналы
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)

    def stretch(channel):
        x_min = np.min(channel)
        x_max = np.max(channel)

        if x_max == x_min:
            return channel.astype(np.uint8)

        result = ((channel - x_min) * (y_max - y_min) /
                  (x_max - x_min)) + y_min

        return np.round(result).astype(np.uint8)

    red_stretched = stretch(red)
    green_stretched = stretch(green)
    blue_stretched = stretch(blue)

    return red_stretched, green_stretched, blue_stretched


def show_linear_contrast_results(red, green, blue,
                                 red_contrast, green_contrast, blue_contrast,
                                 y_min=0, y_max=255):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 3, 1)
    plt.imshow(red)
    plt.title('Исходный RED')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(green)
    plt.title('Исходный GREEN')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(blue)
    plt.title('Исходный BLUE')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(red_contrast)
    plt.title(f'RED после ЛК [{y_min}, {y_max}]')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(green_contrast)
    plt.title(f'GREEN после ЛК [{y_min}, {y_max}]')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(blue_contrast)
    plt.title(f'BLUE после ЛК [{y_min}, {y_max}]')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def solarize_image(img_rgb):
    xmax = 255
    k = 4 / xmax

    solarized_rgb = k * img_rgb * (xmax - img_rgb)
    solarized_rgb = np.clip(solarized_rgb, 0, 255).astype(np.uint8)

    R = solarized_rgb[:, :, 0]
    G = solarized_rgb[:, :, 1]
    B = solarized_rgb[:, :, 2]

    return R, G, B, solarized_rgb


def show_solarization_results(R, G, B, RGB):
    plt.figure(figsize=(12, 6))

    # Отдельные каналы
    plt.subplot(2, 3, 1)
    plt.imshow(R, cmap='gray')
    plt.title('Соляризованный RED')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(G, cmap='gray')
    plt.title('Соляризованный GREEN')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(B, cmap='gray')
    plt.title('Соляризованный BLUE')
    plt.axis('off')

    # RGB изображение
    plt.subplot(2, 3, 4)
    plt.imshow(RGB)
    plt.title('Соляризованное RGB')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def gamma_correction(img_rgb, gamma=1.0):
    height = img_rgb.shape[0]
    width = img_rgb.shape[1]

    img_gamma = np.zeros_like(img_rgb, dtype=np.float32)

    total_brightness = 0.0
    for y in range(height):
        for x in range(width):
            R = img_rgb[y, x, 0]
            G = img_rgb[y, x, 1]
            B = img_rgb[y, x, 2]

            i = 0.299 * R + 0.557 * G + 0.144 * B
            total_brightness += i

            img_gamma[y, x, 0] = 255 * ((R / 255) ** gamma)
            img_gamma[y, x, 1] = 255 * ((G / 255) ** gamma)
            img_gamma[y, x, 2] = 255 * ((B / 255) ** gamma)

    mean_brightness = total_brightness / (height * width)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                if img_gamma[y, x, c] > 255:
                    img_gamma[y, x, c] = 255
                elif img_gamma[y, x, c] < 0:
                    img_gamma[y, x, c] = 0

    img_gamma = img_gamma.astype(np.uint8)

    R_channel = img_gamma[:, :, 0]
    G_channel = img_gamma[:, :, 1]
    B_channel = img_gamma[:, :, 2]

    return R_channel, G_channel, B_channel, img_gamma


def show_gamma_correction_results(R, G, B, RGB, gamma):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(R)
    plt.title(f'RED канал (γ={gamma})')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(G)
    plt.title(f'GREEN канал (γ={gamma})')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(B)
    plt.title(f'BLUE канал (γ={gamma})')
    plt.axis('off')

    # RGB изображение после гамма-коррекции
    plt.subplot(1, 4, 4)
    plt.imshow(RGB)
    plt.title(f'RGB изображение (γ={gamma})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def compute_color_histogram(img_rgb):
    hist_R = [0] * 256
    hist_G = [0] * 256
    hist_B = [0] * 256

    height = img_rgb.shape[0]
    width = img_rgb.shape[1]

    for y in range(height):
        for x in range(width):
            R = int(img_rgb[y, x, 0])
            G = int(img_rgb[y, x, 1])
            B = int(img_rgb[y, x, 2])

            hist_R[R] += 1
            hist_G[G] += 1
            hist_B[B] += 1

    return hist_R, hist_G, hist_B


def show_color_histogram(hist_R, hist_G, hist_B, title="Гистограмма каналов RGB"):
    plt.figure(figsize=(10, 5))
    plt.plot(hist_R, color='red', label='Red')
    plt.plot(hist_G, color='green', label='Green')
    plt.plot(hist_B, color='blue', label='Blue')

    plt.title(title)
    plt.xlabel("Яркость (0-255)")
    plt.ylabel("Количество пикселей")
    plt.legend()
    plt.grid(True)
    plt.show()


def gray_correction(img_rgb):
    img_corrected = img_rgb.astype(np.float32)

    height, width = img_rgb.shape[:2]
    num_pixels = height * width

    sum_R = 0.0
    sum_G = 0.0
    sum_B = 0.0
    for y in range(height):
        for x in range(width):
            sum_R += img_rgb[y, x, 0]
            sum_G += img_rgb[y, x, 1]
            sum_B += img_rgb[y, x, 2]

    mean_R = sum_R / num_pixels
    mean_G = sum_G / num_pixels
    mean_B = sum_B / num_pixels

    mean_gray = (mean_R + mean_G + mean_B) / 3.0

    k_R = mean_gray / mean_R
    k_G = mean_gray / mean_G
    k_B = mean_gray / mean_B

    for y in range(height):
        for x in range(width):
            img_corrected[y, x, 0] = min(max(img_rgb[y, x, 0] * k_R, 0), 255)
            img_corrected[y, x, 1] = min(max(img_rgb[y, x, 1] * k_G, 0), 255)
            img_corrected[y, x, 2] = min(max(img_rgb[y, x, 2] * k_B, 0), 255)

    img_corrected = img_corrected.astype(np.uint8)

    return img_corrected


def show_image_gray_correction(img_original, img_corrected):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 3, 1)
    plt.imshow(img_original)
    plt.title("Исходное RGB")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(img_corrected)
    plt.title("После 'серый мир'")
    plt.axis("off")

    # Отдельные каналы после коррекции
    plt.subplot(2, 3, 4)
    plt.imshow(parse_grayscale(img_corrected[:, :, 0]))
    plt.title("R канал")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(parse_grayscale(img_corrected[:, :, 1]))
    plt.title("G канал")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(parse_grayscale(img_corrected[:, :, 2]))
    plt.title("B канал")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def log_correction(image):
    rows, cols = image.shape[:2]
    log_image = np.zeros_like(image, dtype=np.uint8)

    R_max = int(np.max(image))
    if R_max == 0:  #
        c = 1
    else:
        c = 255 / math.log(1 + R_max)

    if len(image.shape) == 3:
        for i in range(rows):
            for j in range(cols):
                for k in range(3):
                    r = int(image[i, j, k])
                    log_image[i, j, k] = min(int(c * math.log(1 + r)), 255)
    else:
        for i in range(rows):
            for j in range(cols):
                r = int(image[i, j])
                log_image[i, j] = min(int(c * math.log(1 + r)), 255)

    return log_image


def show_image_log_correction(img_original, img_corrected):
    plt.figure(figsize=(10, 5))

    # Исходное RGB изображение
    plt.subplot(2, 3, 1)
    plt.imshow(img_original)
    plt.title("Исходное RGB")
    plt.axis("off")

    # Лог-коррекция RGB
    plt.subplot(2, 3, 2)
    plt.imshow(img_corrected)
    plt.title("После логарифмической коррекции")
    plt.axis("off")

    # Каналы после лог-коррекции
    plt.subplot(2, 3, 4)
    plt.imshow(parse_grayscale(img_corrected[:, :, 0]))
    plt.title("R канал")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(parse_grayscale(img_corrected[:, :, 1]))
    plt.title("G канал")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(parse_grayscale(img_corrected[:, :, 2]))
    plt.title("B канал")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def saw_contrast_color(img_rgb, mode: int, N: int = 3):
    """
    Параметры:
        img_rgb — входное RGB изображение (uint8)
        mode — тип преобразования (1, 2, 3, 4)
        N — число периодов (для mode=4)
    """

    img = img_rgb.astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    def transform(channel):

        y = np.zeros_like(channel)

        if mode == 1:
            # Тип 1
            x1, x2 = 80, 200
            mask = (channel >= x1) & (channel <= x2)
            y[mask] = (channel[mask] - x1) * 255.0 / (x2 - x1)
            y[channel > x2] = 100

        elif mode == 2:
            # Тип 2
            x1 = 100
            mask = channel <= x1
            y[mask] = channel[mask] * 255.0 / x1
            y[channel > x1] = 255

        elif mode == 3:
            # Тип 3
            x1, x2 = 50, 180
            mask = (channel >= x1) & (channel <= x2)
            y[mask] = (channel[mask] - x1) * 255.0 / (x2 - x1)

        elif mode == 4:
            # Тип 4
            if N <= 0:
                raise ValueError("N должно быть > 0")
            L = 256
            period = L / N
            y = (channel % period) * (255.0 / period)

        else:
            raise ValueError("mode должен быть 1, 2, 3 или 4")

        return np.clip(y, 0, 255)

    R_new = transform(R)
    G_new = transform(G)
    B_new = transform(B)

    img_new = np.stack([R_new, G_new, B_new], axis=2).astype(np.uint8)

    return R_new.astype(np.uint8), \
        G_new.astype(np.uint8), \
        B_new.astype(np.uint8), \
        img_new


def show_image_saw_contrast(img_original, mode: int, N: int = 3):

    R, G, B, img_corrected = saw_contrast_color(img_original, mode, N)

    plt.figure(figsize=(12, 6))

    # Исходное изображение
    plt.subplot(2, 3, 1)
    plt.imshow(img_original)
    plt.title("Исходное RGB")
    plt.axis("off")

    # После контрастирования
    plt.subplot(2, 3, 2)
    plt.imshow(img_corrected)
    plt.title(f"Пилообразное контрастирование (тип {mode})")
    plt.axis("off")

    # R канал
    plt.subplot(2, 3, 4)
    plt.imshow(R, cmap='gray')
    plt.title("R канал")
    plt.axis("off")

    # G канал
    plt.subplot(2, 3, 5)
    plt.imshow(G, cmap='gray')
    plt.title("G канал")
    plt.axis("off")

    # B канал
    plt.subplot(2, 3, 6)
    plt.imshow(B, cmap='gray')
    plt.title("B канал")
    plt.axis("off")

    plt.tight_layout()
    plt.show()