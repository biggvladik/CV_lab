import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

    h = len(img)
    w = len(img[0])

    and_img = [[0] * w for _ in range(h)]
    or_img = [[0] * w for _ in range(h)]
    xor_img = [[0] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            p = img[i][j]

            # AND
            if p != 0 and const != 0:
                and_img[i][j] = 255
            else:
                and_img[i][j] = 0

            # OR
            if p != 0 or const != 0:
                or_img[i][j] = 255
            else:
                or_img[i][j] = 0

            # XOR
            if (p != 0 and const == 0) or (p == 0 and const != 0):
                xor_img[i][j] = 255
            else:
                xor_img[i][j] = 0

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
    # получаем размеры
    h, w = img.shape[:2]
    c = 1 if len(img.shape) == 2 else img.shape[2]

    # создаём результат
    if c == 1:
        result = [[0]*w for _ in range(h)]
    else:
        result = [[[0]*c for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if c == 1:  # grayscale
                val = int(img[i, j])
                bits = list(f"{val:08b}")
                bits = ['0' if b == '1' else '1' for b in bits]
                result[i][j] = int("".join(bits), 2)
            else:  # цветное изображение
                for k in range(c):
                    val = int(img[i, j, k])
                    bits = list(f"{val:08b}")
                    bits = ['0' if b == '1' else '1' for b in bits]
                    result[i][j][k] = int("".join(bits), 2)

    return result



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
    h, w = img.shape[:2]
    c = 1 if len(img.shape) == 2 else img.shape[2]

    # создаём результаты
    if c == 1:
        eq = [[0]*w for _ in range(h)]
        gt = [[0]*w for _ in range(h)]
        lt = [[0]*w for _ in range(h)]
    else:
        eq = [[[0]*c for _ in range(w)] for _ in range(h)]
        gt = [[[0]*c for _ in range(w)] for _ in range(h)]
        lt = [[[0]*c for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if c == 1:
                p = int(img[i, j])
                eq[i][j] = 255 if p == const else 0
                gt[i][j] = 255 if p > const else 0
                lt[i][j] = 255 if p < const else 0
            else:
                for k in range(c):
                    p = int(img[i, j, k])
                    eq[i][j][k] = 255 if p == const else 0
                    gt[i][j][k] = 255 if p > const else 0
                    lt[i][j][k] = 255 if p < const else 0

    # отображение
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    if c == 1:
        axs[0].imshow(eq, cmap="gray")
        axs[1].imshow(gt, cmap="gray")
        axs[2].imshow(lt, cmap="gray")
    else:
        axs[0].imshow(eq)
        axs[1].imshow(gt)
        axs[2].imshow(lt)

    axs[0].set_title("=")
    axs[1].set_title(">")
    axs[2].set_title("<")

    for ax in axs:
        ax.axis("off")

    plt.show()


# =========================================================
# Сравнение двух изображений
# =========================================================
def compare_images(img1, img2):
    # приводим второе изображение к размеру первого
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    h, w = img1.shape[:2]
    c = 1 if len(img1.shape) == 2 else img1.shape[2]

    # создаём результаты
    if c == 1:
        eq = [[0]*w for _ in range(h)]
        gt = [[0]*w for _ in range(h)]
        lt = [[0]*w for _ in range(h)]
    else:
        eq = [[[0]*c for _ in range(w)] for _ in range(h)]
        gt = [[[0]*c for _ in range(w)] for _ in range(h)]
        lt = [[[0]*c for _ in range(w)] for _ in range(h)]

    # сравнение пиксель за пикселем
    for i in range(h):
        for j in range(w):
            if c == 1:  # grayscale
                p1 = int(img1[i, j])
                p2 = int(img2[i, j])
                eq[i][j] = 255 if p1 == p2 else 0
                gt[i][j] = 255 if p1 > p2 else 0
                lt[i][j] = 255 if p1 < p2 else 0
            else:  # цветное изображение
                for k in range(c):
                    p1 = int(img1[i, j, k])
                    p2 = int(img2[i, j, k])
                    eq[i][j][k] = 255 if p1 == p2 else 0
                    gt[i][j][k] = 255 if p1 > p2 else 0
                    lt[i][j][k] = 255 if p1 < p2 else 0

    # отображение
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    if c == 1:
        axs[0].imshow(eq, cmap="gray")
        axs[1].imshow(gt, cmap="gray")
        axs[2].imshow(lt, cmap="gray")
    else:
        axs[0].imshow(eq)
        axs[1].imshow(gt)
        axs[2].imshow(lt)

    axs[0].set_title("=")
    axs[1].set_title(">")
    axs[2].set_title("<")

    for ax in axs:
        ax.axis("off")

    plt.show()


# ============================
# Арифметика между изображениями
# ============================
def arithmetic_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w = img1.shape[:2]
    c = 1 if len(img1.shape) == 2 else img1.shape[2]

    if c == 1:
        add = [[0]*w for _ in range(h)]
        sub = [[0]*w for _ in range(h)]
        mul = [[0]*w for _ in range(h)]
    else:
        add = [[[0]*c for _ in range(w)] for _ in range(h)]
        sub = [[[0]*c for _ in range(w)] for _ in range(h)]
        mul = [[[0]*c for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if c == 1:
                p1, p2 = int(img1[i, j]), int(img2[i, j])
                add[i][j] = min(p1 + p2, 255)
                sub[i][j] = max(p1 - p2, 0)
                mul[i][j] = min(p1 * p2, 255)
            else:
                for k in range(c):
                    p1, p2 = int(img1[i, j, k]), int(img2[i, j, k])
                    add[i][j][k] = min(p1 + p2, 255)
                    sub[i][j][k] = max(p1 - p2, 0)
                    mul[i][j][k] = min(p1 * p2, 255)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(add, cmap="gray" if c==1 else None)
    axs[0].set_title("Add")
    axs[1].imshow(sub, cmap="gray" if c==1 else None)
    axs[1].set_title("Subtract")
    axs[2].imshow(mul, cmap="gray" if c==1 else None)
    axs[2].set_title("Multiply")
    for ax in axs: ax.axis("off")
    plt.show()


# ============================
# Арифметика с константой
# ============================
def arithmetic_const(img, const):
    h, w = img.shape[:2]
    c = 1 if len(img.shape) == 2 else img.shape[2]

    if c == 1:
        add = [[0]*w for _ in range(h)]
        sub = [[0]*w for _ in range(h)]
        mul = [[0]*w for _ in range(h)]
    else:
        add = [[[0]*c for _ in range(w)] for _ in range(h)]
        sub = [[[0]*c for _ in range(w)] for _ in range(h)]
        mul = [[[0]*c for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if c == 1:
                p = int(img[i, j])
                add[i][j] = min(p + const, 255)
                sub[i][j] = max(p - const, 0)
                mul[i][j] = min(p * const, 255)
            else:
                for k in range(c):
                    p = int(img[i, j, k])
                    add[i][j][k] = min(p + const, 255)
                    sub[i][j][k] = max(p - const, 0)
                    mul[i][j][k] = min(p * const, 255)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(add, cmap="gray" if c==1 else None)
    axs[0].set_title("+ const")
    axs[1].imshow(sub, cmap="gray" if c==1 else None)
    axs[1].set_title("- const")
    axs[2].imshow(mul, cmap="gray" if c==1 else None)
    axs[2].set_title("* const")
    for ax in axs: ax.axis("off")
    plt.show()


# ============================
# Деление изображений
# ============================
def divide_images(img1, img2):
    import matplotlib.pyplot as plt
    import cv2

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w = img1.shape[:2]
    c = 1 if len(img1.shape) == 2 else img1.shape[2]

    # создаём результат
    result = [[[0] * c for _ in range(w)] for _ in range(h)] if c > 1 else [[0] * w for _ in range(h)]

    # деление
    if c == 1:
        for i in range(h):
            for j in range(w):
                p1, p2 = float(img1[i, j]), float(img2[i, j])
                if p2 == 0: p2 = 1
                result[i][j] = p1 / p2
        # находим min/max
        min_val = min([min(row) for row in result])
        max_val = max([max(row) for row in result])
        # нормализация
        norm = [[int((result[i][j] - min_val) / (max_val - min_val) * 255) for j in range(w)] for i in range(h)]
    else:
        # цветное изображение
        # делаем нормализацию для каждого канала отдельно
        result = [[[0] * c for _ in range(w)] for _ in range(h)]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    p1 = float(img1[i, j, k])
                    p2 = float(img2[i, j, k])
                    if p2 == 0: p2 = 1
                    result[i][j][k] = p1 / p2
        # нормализация по каждому каналу
        norm = [[[0] * c for _ in range(w)] for _ in range(h)]
        for k in range(c):
            # находим min/max для канала k
            min_val = min([result[i][j][k] for i in range(h) for j in range(w)])
            max_val = max([result[i][j][k] for i in range(h) for j in range(w)])
            if max_val != min_val:
                for i in range(h):
                    for j in range(w):
                        norm[i][j][k] = int((result[i][j][k] - min_val) / (max_val - min_val) * 255)
            else:
                for i in range(h):
                    for j in range(w):
                        norm[i][j][k] = 0

    plt.imshow(norm, cmap="gray" if c == 1 else None)
    plt.title("Division")
    plt.axis("off")
    plt.show()


# ============================
# Нормы между изображениями
# ============================
def image_norms(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w = img1.shape[:2]
    c = 1 if len(img1.shape) == 2 else img1.shape[2]

    c_norm = 0
    l1_norm = 0
    l2_sum = 0

    for i in range(h):
        for j in range(w):
            if c == 1:
                p1, p2 = float(img1[i, j]), float(img2[i, j])
                diff = p1 - p2
                abs_diff = abs(diff)
                c_norm = max(c_norm, abs_diff)
                l1_norm += abs_diff
                l2_sum += diff*diff
            else:
                for k in range(c):
                    p1, p2 = float(img1[i, j, k]), float(img2[i, j, k])
                    diff = p1 - p2
                    abs_diff = abs(diff)
                    c_norm = max(c_norm, abs_diff)
                    l1_norm += abs_diff
                    l2_sum += diff*diff

    l2_norm = math.sqrt(l2_sum)

    print("C norm:", c_norm)
    print("L1 norm:", l1_norm)
    print("L2 norm:", l2_norm)
