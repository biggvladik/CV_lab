from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return np.clip(image @ np.array([0.299, 0.587, 0.114]), 0.0, 1.0)


def gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return np.clip(image + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)


def salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    result = image.copy()
    mask = rng.random(image.shape[:2])
    result[mask < amount / 2] = 0.0
    result[(mask >= amount / 2) & (mask < amount)] = 1.0
    return result


def disk_psf(radius: int = 10) -> np.ndarray:
    if radius < 1:
        raise ValueError("Радиус диска должен быть положительным")
    coordinates = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(coordinates, coordinates)
    psf = ((x ** 2 + y ** 2) <= radius ** 2).astype(np.float64)
    return psf / psf.sum()


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def circular_defocus(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    otf = psf_to_otf(psf, image.shape[:2])
    spectrum = np.fft.fft2(image, axes=(0, 1))
    result = np.fft.ifft2(spectrum * otf[..., None], axes=(0, 1)).real
    return np.clip(result, 0.0, 1.0)


def symmetric_defocus(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    result = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT)
    return np.clip(result, 0.0, 1.0)


def valid_defocus(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    result = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_CONSTANT)
    py, px = psf.shape[0] // 2, psf.shape[1] // 2
    return np.clip(result[py:image.shape[0] - py, px:image.shape[1] - px], 0.0, 1.0)


def edge_taper(image: np.ndarray, psf: np.ndarray, width: int = 30) -> np.ndarray:
    smooth = symmetric_defocus(image, psf)
    height, width_image = image.shape[:2]
    distance_y = np.minimum(np.arange(height), np.arange(height)[::-1])
    distance_x = np.minimum(np.arange(width_image), np.arange(width_image)[::-1])
    weight_y = np.sin(0.5 * np.pi * np.clip(distance_y / width, 0, 1)) ** 2
    weight_x = np.sin(0.5 * np.pi * np.clip(distance_x / width, 0, 1)) ** 2
    weight = (weight_y[:, None] * weight_x[None, :])[..., None]
    return weight * image + (1.0 - weight) * smooth


def gibbs_reduced_defocus(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return circular_defocus(edge_taper(image, psf), psf)


def _spectral_restore(blurred: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(blurred, axes=(0, 1))
    result = np.fft.ifft2(spectrum * transfer[..., None], axes=(0, 1)).real
    return np.clip(result, 0.0, 1.0)


def inverse_filter(blurred: np.ndarray, psf: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    otf = psf_to_otf(psf, blurred.shape[:2])
    safe_otf = np.where(np.abs(otf) >= eps, otf, eps * np.exp(1j * np.angle(otf)))
    return _spectral_restore(blurred, 1.0 / safe_otf)


def pseudo_inverse_filter(
    blurred: np.ndarray,
    psf: np.ndarray,
    cutoff: float = 0.1,
) -> np.ndarray:
    otf = psf_to_otf(psf, blurred.shape[:2])
    fy = np.fft.fftfreq(blurred.shape[0])[:, None]
    fx = np.fft.fftfreq(blurred.shape[1])[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    mask = (radius <= 0.5 * cutoff) & (np.abs(otf) >= 1e-8)
    transfer = np.zeros_like(otf)
    transfer[mask] = 1.0 / otf[mask]
    return _spectral_restore(blurred, transfer)


def tikhonov_filter(
    blurred: np.ndarray,
    psf: np.ndarray,
    alpha: float = 0.003,
    order: int = 1,
) -> np.ndarray:
    otf = psf_to_otf(psf, blurred.shape[:2])
    wy = 2 * np.pi * np.fft.fftfreq(blurred.shape[0])[:, None]
    wx = 2 * np.pi * np.fft.fftfreq(blurred.shape[1])[None, :]
    penalty = (wx ** 2 + wy ** 2) ** order
    transfer = np.conj(otf) / (np.abs(otf) ** 2 + alpha * penalty)
    return _spectral_restore(blurred, transfer)


def wiener_filter(
    blurred: np.ndarray,
    psf: np.ndarray,
    noise_to_signal: float = 0.003,
) -> np.ndarray:
    otf = psf_to_otf(psf, blurred.shape[:2])
    transfer = np.conj(otf) / (np.abs(otf) ** 2 + noise_to_signal)
    return _spectral_restore(blurred, transfer)


def restore_all(
    blurred: np.ndarray,
    psf: np.ndarray,
    noisy: bool = False,
) -> dict[str, np.ndarray]:
    return {
        "Обратный ПФ": inverse_filter(blurred, psf),
        "Псевдоинверсия": pseudo_inverse_filter(blurred, psf, 0.065 if noisy else 0.11),
        "Тихонов (ПФ)": tikhonov_filter(blurred, psf, 0.018 if noisy else 0.0005),
        "Фильтр Винера": wiener_filter(blurred, psf, 0.018 if noisy else 0.0005),
    }
