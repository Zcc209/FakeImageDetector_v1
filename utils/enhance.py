from typing import Any

import cv2
import numpy as np


def _upscale_with_roi_hint(img_rgb: np.ndarray, scale: float = 1.5) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    # Keep center content a bit tighter, then upscale (flowchart: ROI crop + zoom).
    x_margin = int(w * 0.05)
    y_margin = int(h * 0.05)
    roi = img_rgb[y_margin:max(y_margin + 1, h - y_margin), x_margin:max(x_margin + 1, w - x_margin)]
    up_w = max(w, int(roi.shape[1] * scale))
    up_h = max(h, int(roi.shape[0] * scale))
    return cv2.resize(roi, (up_w, up_h), interpolation=cv2.INTER_CUBIC)


def _denoise_blur(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, 7, 7, 7, 21)


def _adjust_exposure(img_rgb: np.ndarray, brighten: bool) -> np.ndarray:
    alpha = 1.15 if brighten else 0.9  # contrast
    beta = 18 if brighten else -16  # brightness
    return cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)


def _deblock_and_denoise(img_rgb: np.ndarray) -> np.ndarray:
    bilateral = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=40, sigmaSpace=40)
    return cv2.fastNlMeansDenoisingColored(bilateral, None, 6, 6, 7, 21)


def enhance_by_quality_reasons(img_rgb: np.ndarray, reasons: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Apply targeted enhancement from quality-gate reasons.
    Returns enhanced image + applied action logs.
    """
    out = img_rgb.copy()
    actions: list[dict[str, Any]] = []

    if "low_resolution" in reasons:
        out = _upscale_with_roi_hint(out, scale=1.6)
        actions.append({"code": "low_resolution", "action": "roi_crop_upscale"})
    if "too_blurry" in reasons:
        out = _denoise_blur(out)
        actions.append({"code": "too_blurry", "action": "fastNlMeansDenoisingColored"})
    if "too_dark" in reasons:
        out = _adjust_exposure(out, brighten=True)
        actions.append({"code": "too_dark", "action": "exposure_contrast_boost"})
    if "too_bright" in reasons:
        out = _adjust_exposure(out, brighten=False)
        actions.append({"code": "too_bright", "action": "exposure_contrast_reduce"})
    if "too_compressed" in reasons:
        out = _deblock_and_denoise(out)
        actions.append({"code": "too_compressed", "action": "deblocking_denoise"})

    return out, actions
