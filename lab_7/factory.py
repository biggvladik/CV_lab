
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return np.clip(image @ np.array([0.299, 0.587, 0.114]), 0.0, 1.0)


def gaussian_noise(
    image: np.ndarray, mean: float = 0.0, sigma: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return np.clip(image + rng.normal(mean, sigma, image.shape), 0.0, 1.0)


def salt_pepper_noise(
    image: np.ndarray, amount: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    result = image.copy()
    mask = rng.random(image.shape[:2])
    result[mask < amount / 2] = 0.0
    result[(mask >= amount / 2) & (mask < amount)] = 1.0
    return result


def horizontal_motion_psf(length: int = 21) -> np.ndarray:
    if length < 1 or length % 2 == 0:
        raise ValueError("Длина смазывания должна быть положительным нечетным числом")
    return np.ones((1, length), dtype=np.float64) / length


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def circular_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    otf = psf_to_otf(psf, image.shape[:2])
    result = np.fft.ifft2(np.fft.fft2(image, axes=(0, 1)) * otf[..., None], axes=(0, 1)).real
    return np.clip(result, 0.0, 1.0)


def spatial_blur(image: np.ndarray, psf: np.ndarray, boundary: str = "reflect") -> np.ndarray:
    modes = {"constant": "constant", "reflect": "reflect", "wrap": "wrap"}
    if boundary not in (*modes, "antireflect"):
        raise ValueError(f"Неизвестное граничное условие: {boundary}")
    kh, kw = psf.shape
    py, px = kh // 2, kw // 2
    if boundary == "antireflect":
        if py:
            top = 2 * image[:1] - image[1:py + 1][::-1]
            bottom = 2 * image[-1:] - image[-py - 1:-1][::-1]
            padded = np.concatenate((top, image, bottom), axis=0)
        else:
            padded = image
        if px:
            left = 2 * padded[:, :1] - padded[:, 1:px + 1][:, ::-1]
            right = 2 * padded[:, -1:] - padded[:, -px - 1:-1][:, ::-1]
            padded = np.concatenate((left, padded, right), axis=1)
    else:
        pad_spec = ((py, py), (px, px), (0, 0))
        padded = np.pad(image, pad_spec, mode=modes[boundary])
    result = np.zeros_like(image)
    for y in range(kh):
        for x in range(kw):
            result += psf[y, x] * padded[y:y + image.shape[0], x:x + image.shape[1]]
    return np.clip(result, 0.0, 1.0)


def valid_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    kh, kw = psf.shape
    out_h, out_w = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    result = np.zeros((out_h, out_w, image.shape[2]), dtype=np.float64)
    for y in range(kh):
        for x in range(kw):
            result += psf[y, x] * image[y:y + out_h, x:x + out_w]
    return np.clip(result, 0.0, 1.0)


def gibbs_reduced_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    delta = psf.shape[1] - 1
    extended = np.pad(image, ((0, 0), (delta, delta), (0, 0)), mode="constant")
    return valid_blur(extended, psf)


def _spectral_restore(blurred: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(blurred, axes=(0, 1))
    result = np.fft.ifft2(spectrum * transfer[..., None], axes=(0, 1)).real
    return np.clip(result, 0.0, 1.0)


def inverse_filter(blurred: np.ndarray, psf: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    h = psf_to_otf(psf, blurred.shape[:2])
    safe_h = np.where(np.abs(h) >= eps, h, eps * np.exp(1j * np.angle(h)))
    return _spectral_restore(blurred, 1.0 / safe_h)


def pseudo_inverse_filter(blurred: np.ndarray, psf: np.ndarray, cutoff: float = 0.65) -> np.ndarray:
    h = psf_to_otf(psf, blurred.shape[:2])
    transfer = np.zeros_like(h)
    frequency_x = np.abs(np.fft.fftfreq(blurred.shape[1]))
    mask = (frequency_x[None, :] <= 0.5 * cutoff) & (np.abs(h) >= 1e-8)
    transfer[mask] = 1.0 / h[mask]
    return _spectral_restore(blurred, transfer)


def tikhonov_fourier(
    blurred: np.ndarray, psf: np.ndarray, alpha: float = 0.003, order: int = 1,
) -> np.ndarray:
    h = psf_to_otf(psf, blurred.shape[:2])
    omega = 2 * np.pi * np.fft.fftfreq(blurred.shape[1])
    penalty = np.abs(omega)[None, :] ** (2 * order)
    transfer = np.conj(h) / (np.abs(h) ** 2 + alpha * penalty)
    return _spectral_restore(blurred, transfer)


def tikhonov_finite_sum(
    blurred_valid: np.ndarray, psf: np.ndarray, output_width: int,
    alpha: float = 0.003,
) -> np.ndarray:
    length = psf.shape[1]
    rows = output_width - length + 1
    if blurred_valid.shape[1] != rows:
        raise ValueError("Ширина усеченного смаза должна быть n-Δ")
    a = np.zeros((rows, output_width), dtype=np.float64)
    for i in range(rows):
        a[i, i:i + length] = psf[0]
    normal = a.T @ a + alpha * np.eye(output_width)
    data = blurred_valid.transpose(1, 0, 2).reshape(rows, -1)
    solution = np.linalg.solve(normal, a.T @ data)
    result = solution.reshape(output_width, blurred_valid.shape[0], 3).transpose(1, 0, 2)
    return np.clip(result, 0.0, 1.0)


def wiener_filter(blurred: np.ndarray, psf: np.ndarray, noise_to_signal: float = 0.003) -> np.ndarray:
    h = psf_to_otf(psf, blurred.shape[:2])
    transfer = np.conj(h) / (np.abs(h) ** 2 + noise_to_signal)
    return _spectral_restore(blurred, transfer)


def restore_all(
    blurred: np.ndarray, blurred_valid: np.ndarray, psf: np.ndarray,
    output_width: int, noisy: bool = False,
) -> dict[str, np.ndarray]:
    return {
        "Обратный ПФ": inverse_filter(blurred, psf),
        "Псевдоинверсия": pseudo_inverse_filter(blurred, psf, 0.06 if noisy else 0.09),
        "Тихонов (ПФ, p=1)": tikhonov_fourier(blurred, psf, 0.012 if noisy else 0.0004, order=1),
        "Тихонов (конечная сумма)": tikhonov_finite_sum(
            blurred_valid, psf, output_width, 0.012 if noisy else 0.0004
        ),
        "Фильтр Винера": wiener_filter(blurred, psf, 0.012 if noisy else 0.0004),
    }


