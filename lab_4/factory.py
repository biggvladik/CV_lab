import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Функция чтения изображения
# Читает изображение с диска и переводит его из BGR в RGB
# =========================================================
def read_image(path):
    img = cv2.imread(path)

    # Проверяем, существует ли файл
    if img is None:
        raise ValueError(f"Image not found: {path}")

    # OpenCV читает изображение в формате BGR,
    # переводим его в RGB для корректного отображения
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# =========================================================
# Отображение цветного изображения:
# RGB + каналы R G B + grayscale
# =========================================================
def show_channels(img, title="Image"):
    # Разделяем каналы
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Переводим в оттенки серого
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

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


# =========================================================
# Отображение черно-белого изображения
# =========================================================
def show_bw(img, title="Image"):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================================================
# Перевод изображения в оттенки серого
# =========================================================
def to_gray(img):
    if len(img.shape) == 3:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        img = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    return img


# =========================================================
# Побитовая бинарная операция между двумя изображениями
# (AND OR XOR)
# =========================================================
def binary_operation(img1, img2, op):
    h, w = img1.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            a = img1[i, j]
            b = img2[i, j]

            a_bin = format(a, "08b")
            b_bin = format(b, "08b")

            res_bin = ""

            for k in range(8):

                bit_a = int(a_bin[k])
                bit_b = int(b_bin[k])

                if op == "and":
                    res_bin += str(bit_a & bit_b)

                elif op == "or":
                    res_bin += str(bit_a | bit_b)

                elif op == "xor":
                    res_bin += str((bit_a + bit_b) % 2)

            result[i, j] = int(res_bin, 2)

    return result


# =========================================================
# Выполнение логических операций AND OR XOR
# =========================================================
def logical_operations(img1, img2, title="Logical operations"):
    img1 = to_gray(img1)
    img2 = to_gray(img2)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    and_img = binary_operation(img1, img2, "and")
    or_img = binary_operation(img1, img2, "or")
    xor_img = binary_operation(img1, img2, "xor")

    fig, axs = plt.subplots(1, 5, figsize=(16, 4))
    fig.suptitle(title)

    axs[0].imshow(img1, cmap="gray")
    axs[0].set_title("Image 1")

    axs[1].imshow(img2, cmap="gray")
    axs[1].set_title("Image 2")

    axs[2].imshow(and_img, cmap="gray")
    axs[2].set_title("AND")

    axs[3].imshow(or_img, cmap="gray")
    axs[3].set_title("OR")

    axs[4].imshow(xor_img, cmap="gray")
    axs[4].set_title("XOR")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# Унарные логические операции с константой
# =========================================================
def unary_logical_with_const(img, const):
    img = to_gray(img)

    and_img = img & const
    or_img = img | const
    xor_img = img ^ const

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original")

    axs[1].imshow(and_img, cmap="gray")
    axs[1].set_title("AND const")

    axs[2].imshow(or_img, cmap="gray")
    axs[2].set_title("OR const")

    axs[3].imshow(xor_img, cmap="gray")
    axs[3].set_title("XOR const")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Логическое отрицание изображения
# =========================================================
def logical_not(img):
    return 255 - img


# =========================================================
# Показ отрицания для цветного и черно-белого изображения
# =========================================================
def show_not(color_img, bw_img):
    not_color = logical_not(color_img)
    not_bw = logical_not(to_gray(bw_img))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(color_img)
    axs[0, 0].set_title("Color")

    axs[0, 1].imshow(not_color)
    axs[0, 1].set_title("NOT Color")

    axs[1, 0].imshow(to_gray(bw_img), cmap="gray")
    axs[1, 0].set_title("BW")

    axs[1, 1].imshow(not_bw, cmap="gray")
    axs[1, 1].set_title("NOT BW")

    for ax in axs.ravel():
        ax.axis("off")

    plt.show()


# =========================================================
# Сравнение изображения с константой
# =========================================================
def compare_with_const(img, const):
    eq = (img == const) * 255
    gt = (img > const) * 255
    lt = (img < const) * 255

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(eq.astype(np.uint8))
    axs[0].set_title("=")

    axs[1].imshow(gt.astype(np.uint8))
    axs[1].set_title(">")

    axs[2].imshow(lt.astype(np.uint8))
    axs[2].set_title("<")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Сравнение двух изображений
# =========================================================
def compare_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    eq = (img1 == img2) * 255
    gt = (img1 > img2) * 255
    lt = (img1 < img2) * 255

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(eq.astype(np.uint8))
    axs[0].set_title("=")

    axs[1].imshow(gt.astype(np.uint8))
    axs[1].set_title(">")

    axs[2].imshow(lt.astype(np.uint8))
    axs[2].set_title("<")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Арифметические операции между изображениями
# =========================================================
def arithmetic_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    add = cv2.add(img1, img2)
    sub = cv2.subtract(img1, img2)
    mul = cv2.multiply(img1, img2)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(add)
    axs[0].set_title("Add")

    axs[1].imshow(sub)
    axs[1].set_title("Subtract")

    axs[2].imshow(mul)
    axs[2].set_title("Multiply")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Арифметические операции с константой
# =========================================================
def arithmetic_const(img, const):
    add = cv2.add(img, const)
    sub = cv2.subtract(img, const)
    mul = cv2.multiply(img, const)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(add)
    axs[0].set_title("+ const")

    axs[1].imshow(sub)
    axs[1].set_title("- const")

    axs[2].imshow(mul)
    axs[2].set_title("* const")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Деление изображений
# =========================================================
def divide_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img2[img2 == 0] = 1

    result = img1 / img2

    min_val = result.min()
    max_val = result.max()

    if max_val != min_val:
        result = (result - min_val) / (max_val - min_val) * 255
    else:
        result = np.zeros_like(result)

    result = result.astype(np.uint8)

    plt.imshow(result, cmap="gray")
    plt.title("Division")
    plt.axis("off")
    plt.show()


def divide_images_gray(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img2[img2 == 0] = 1

    result = img1 / img2

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)

    plt.imshow(result, cmap="gray")
    plt.title("Division")
    plt.axis("off")
    plt.show()


# =========================================================
# Вычисление норм между изображениями
# =========================================================
def image_norms(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = img1.astype(np.float32) - img2.astype(np.float32)

    c_norm = np.max(np.abs(diff))
    l1_norm = np.sum(np.abs(diff))
    l2_norm = np.sqrt(np.sum(diff ** 2))

    print("C norm:", c_norm)
    print("L1 norm:", l1_norm)
    print("L2 norm:", l2_norm)
