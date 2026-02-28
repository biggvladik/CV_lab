from factory import *

img = read_image("test.jpg")

# 1–3
show_channels(img, "Original")

# 4–5 Масштабирование
show_channels(scale_image(img, 0.1), "Scale alpha < 1")
show_channels(scale_image(img, 2), "Scale alpha > 1")

# 6–7 Поворот вокруг начала
show_channels(rotate_origin(img, 20), "Rotate around origin")

# 8–9 Поворот вокруг центра
show_channels(rotate_center(img, 90), "Rotate around center")

# 10–11 Отражение горизонталь
show_channels(flip_horizontal(img), "Flip horizontal")

# 12–13 Отражение вертикаль
show_channels(flip_vertical(img), "Flip vertical")

# 14–15 Скос
show_channels(shear_image(img, sx=1.3, sy=0.5, shx=0.2, shy=1.1), "Shear")