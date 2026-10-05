from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .factory import (
        circular_defocus,
        disk_psf,
        gaussian_noise,
        gibbs_reduced_defocus,
        read_image,
        restore_all,
        salt_pepper_noise,
        symmetric_defocus,
        to_grayscale,
        valid_defocus,
    )
except ImportError:
    from factory import (
        circular_defocus,
        disk_psf,
        gaussian_noise,
        gibbs_reduced_defocus,
        read_image,
        restore_all,
        salt_pepper_noise,
        symmetric_defocus,
        to_grayscale,
        valid_defocus,
    )


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "image.jpg"
RESULTS = ROOT / "results"


def save_figure(
    filename: str,
    title: str,
    images: list[tuple[str, object]],
    columns: int = 3,
) -> None:
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis, (label, image) in zip(axes.flat, images):
        axis.imshow(image, cmap="gray" if getattr(image, "ndim", 3) == 2 else None, vmin=0, vmax=1)
        axis.set_title(label, pad=10)
        axis.axis("off")
    for axis in axes.flat[len(images):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94), h_pad=2.8, w_pad=1.2)
    fig.savefig(RESULTS / filename, dpi=150, bbox_inches="tight")
    plt.show(block=True)
    plt.close(fig)


def truncation_variants(image: np.ndarray, psf: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("Симметричные границы", symmetric_defocus(image, psf)),
        ("Усечение valid", valid_defocus(image, psf)),
        ("Подавление эффекта Гиббса", gibbs_reduced_defocus(image, psf)),
    ]


def restoration_figure(
    filename: str,
    case: str,
    source: np.ndarray,
    blurred: np.ndarray,
    restored: dict[str, np.ndarray],
) -> None:
    images = [("Исходное", source), ("Дефокусированное", blurred)] + list(restored.items())
    save_figure(filename, f"Восстановление: {case}", images, columns=3)


def methods_comparison(
    clean: dict[str, np.ndarray],
    gaussian: dict[str, np.ndarray],
    salt_pepper: dict[str, np.ndarray],
) -> None:
    images = []
    for method in clean:
        images.extend([
            (f"{method}\nбез шума", clean[method]),
            (f"{method}\nГауссов шум", gaussian[method]),
            (f"{method}\nсоль и перец", salt_pepper[method]),
        ])
    save_figure(
        "12_methods_comparison.png",
        "Сравнение восстановления дефокусированных изображений",
        images,
        columns=3,
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    image = read_image(IMAGE_PATH)
    rng = np.random.default_rng(8)

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
        "02_noisy_images.png",
        "Зашумленные цветные изображения",
        [
            ("Исходное", image),
            ("Гауссов шум, σ=0.04", gaussian),
            ("Соль и перец, 4%", salt_pepper),
        ],
        columns=3,
    )

    psf = disk_psf(radius=10)
    defocused = circular_defocus(image, psf)
    save_figure(
        "03_defocus.png",
        "Моделирование дефокусирования однородным диском ρ=10",
        [("Исходное", image), ("PSF", psf / psf.max()), ("Дефокусированное", defocused)],
        columns=3,
    )

    save_figure(
        "04_truncation_and_gibbs.png",
        "Усечение и понижение эффекта Гиббса",
        truncation_variants(image, psf),
        columns=3,
    )

    restored_clean = restore_all(defocused, psf, noisy=False)
    restoration_figure(
        "05_restoration_clean.png", "без шума", image, defocused, restored_clean
    )

    defocused_gaussian = circular_defocus(gaussian, psf)
    save_figure(
        "06_gaussian_defocus.png",
        "Дефокусирование изображения с Гауссовым шумом",
        [("Зашумленное", gaussian), ("Дефокусированное", defocused_gaussian)],
        columns=2,
    )
    save_figure(
        "07_gaussian_truncation_and_gibbs.png",
        "Гауссов шум: усечение и понижение эффекта Гиббса",
        [("До дефокусирования", gaussian)] + truncation_variants(gaussian, psf),
        columns=2,
    )
    restored_gaussian = restore_all(defocused_gaussian, psf, noisy=True)
    restoration_figure(
        "08_restoration_gaussian.png",
        "Гауссов шум",
        gaussian,
        defocused_gaussian,
        restored_gaussian,
    )

    defocused_sp = circular_defocus(salt_pepper, psf)
    save_figure(
        "09_salt_pepper_defocus.png",
        "Дефокусирование изображения с шумом соль и перец",
        [("Зашумленное", salt_pepper), ("Дефокусированное", defocused_sp)],
        columns=2,
    )
    save_figure(
        "10_salt_pepper_truncation_and_gibbs.png",
        "Соль и перец: усечение и понижение эффекта Гиббса",
        [("До дефокусирования", salt_pepper)] + truncation_variants(salt_pepper, psf),
        columns=2,
    )
    restored_sp = restore_all(defocused_sp, psf, noisy=True)
    restoration_figure(
        "11_restoration_salt_pepper.png",
        "шум соль и перец",
        salt_pepper,
        defocused_sp,
        restored_sp,
    )

    methods_comparison(restored_clean, restored_gaussian, restored_sp)


if __name__ == "__main__":
    main()
