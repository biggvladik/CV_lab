

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .factory import (
        circular_blur, gaussian_noise, gibbs_reduced_blur, horizontal_motion_psf,
        read_image, restore_all, salt_pepper_noise,
        spatial_blur, to_grayscale, valid_blur,
    )
except ImportError:
    from factory import (
        circular_blur, gaussian_noise, gibbs_reduced_blur, horizontal_motion_psf,
        read_image, restore_all, salt_pepper_noise,
        spatial_blur, to_grayscale, valid_blur,
    )


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "pictuers.jpg"
RESULTS = ROOT / "results"


def save_figure(name: str, title: str, images: list[tuple[str, object]], columns: int = 3) -> None:
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for ax, (label, image) in zip(axes.flat, images):
        ax.imshow(image, cmap="gray" if getattr(image, "ndim", 3) == 2 else None, vmin=0, vmax=1)
        ax.set_title(label, pad=10)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94), h_pad=2.8, w_pad=1.2)
    fig.savefig(RESULTS / name, dpi=150, bbox_inches="tight")
    plt.show(block=True)
    plt.close(fig)


def restoration_figure(filename: str, case: str, original, blurred, restored) -> None:
    items = [("Исходное", original), ("Смазанное", blurred)] + list(restored.items())
    save_figure(filename, f"Восстановление: {case}", items, columns=3)


def boundary_variants(image, psf):
    return [
        ("нулевое (Дирихле)", spatial_blur(image, psf, "constant")),
        ("симметричное (рефлективное)", spatial_blur(image, psf, "reflect")),
        ("периодическое", spatial_blur(image, psf, "wrap")),
        ("антирефлективное", spatial_blur(image, psf, "antireflect")),
        ("valid (усечение)", valid_blur(image, psf)),
        ("понижение эффекта Гиббса", gibbs_reduced_blur(image, psf)),
    ]


def methods_comparison(filename: str, clean, gaussian, salt_pepper) -> None:
    items = []
    for method in clean:
        items.extend([
            (f"{method}\nбез шума", clean[method]),
            (f"{method}\nГауссов шум", gaussian[method]),
            (f"{method}\nсоль и перец", salt_pepper[method]),
        ])
    save_figure(
        filename,
        "Сравнение методов восстановления с наличием и отсутствием шума",
        items,
        columns=3,
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    image = read_image(IMAGE_PATH)
    rng = np.random.default_rng(7)

    channels = [
        ("Исходное RGB", image),
        ("Красный канал", image[:, :, 0]),
        ("Зеленый канал", image[:, :, 1]),
        ("Синий канал", image[:, :, 2]),
        ("Grayscale", to_grayscale(image)),
    ]
    save_figure("01_channels_and_grayscale.png", "Цветовые каналы", channels, columns=3)

    gaussian = gaussian_noise(image, sigma=0.04, rng=rng)
    salt_pepper = salt_pepper_noise(image, amount=0.04, rng=rng)
    save_figure(
        "02_noisy_images.png", "Зашумленные цветные изображения",
        [("Исходное", image), ("Гауссов шум, σ=0.04", gaussian),
         ("Соль и перец, 4%", salt_pepper)], columns=3,
    )

    psf = horizontal_motion_psf(21)

    blurred = circular_blur(image, psf)
    save_figure(
        "03_horizontal_blur.png", "Горизонтальное смазывание (длина PSF = 21)",
        [("Исходное", image), ("Смазанное", blurred)], columns=2,
    )

    save_figure(
        "04_boundary_conditions.png", "Типы граничных условий",
        boundary_variants(image, psf), columns=3,
    )

    blurred_valid = valid_blur(image, psf)
    restored_clean = restore_all(blurred, blurred_valid, psf, image.shape[1], noisy=False)
    restoration_figure("05_restoration_clean.png", "без шума", image, blurred, restored_clean)

    blurred_gaussian = circular_blur(gaussian, psf)
    save_figure(
        "06_gaussian_horizontal_blur.png",
        "Горизонтальный смаз изображения с Гауссовым шумом",
        [("Зашумленное исходное", gaussian), ("Смазанное", blurred_gaussian)],
        columns=2,
    )
    save_figure(
        "07_gaussian_steps_8_9.png",
        "Шаг 12: повтор шагов 8–9 для изображения с Гауссовым шумом",
        [("Гауссов шум (до смазывания)", gaussian)] + boundary_variants(gaussian, psf),
        columns=3,
    )
    valid_gaussian = valid_blur(gaussian, psf)
    restored_gaussian = restore_all(
        blurred_gaussian, valid_gaussian, psf, image.shape[1], noisy=True
    )
    restoration_figure(
        "08_restoration_gaussian.png", "Гауссов шум", gaussian,
        blurred_gaussian, restored_gaussian,
    )

    blurred_sp = circular_blur(salt_pepper, psf)
    save_figure(
        "09_salt_pepper_horizontal_blur.png",
        "Горизонтальный смаз изображения с шумом «соль и перец»",
        [("Зашумленное исходное", salt_pepper), ("Смазанное", blurred_sp)],
        columns=2,
    )
    save_figure(
        "10_salt_pepper_boundary_conditions.png",
        "Соль и перец: граничные условия, усечение и эффект Гиббса",
        boundary_variants(salt_pepper, psf),
        columns=3,
    )
    valid_sp = valid_blur(salt_pepper, psf)
    restored_sp = restore_all(blurred_sp, valid_sp, psf, image.shape[1], noisy=True)
    restoration_figure(
        "11_restoration_salt_pepper.png", "шум «соль и перец»", salt_pepper,
        blurred_sp, restored_sp,
    )

    methods_comparison(
        "12_methods_comparison.png",
        restored_clean,
        restored_gaussian,
        restored_sp,
    )

if __name__ == "__main__":
    main()
