# src/core/quality_gate.py
import cv2
import numpy as np

_DEFAULT_THRESHOLDS = {
    "min_width": 150,
    "min_height": 150,
    "min_blur_score": 80.0,
    "min_brightness": 30.0,
    "max_brightness": 240.0,
    # JPEG block artifact heuristic: boundary mean / non-boundary mean.
    # Larger value means stronger block edges, often from heavy compression.
    "max_blockiness_ratio": 1.8,
    "enable_compression_check": True,
}


def _compute_blockiness(gray: np.ndarray, block_size: int = 8) -> dict:
    """
    Estimate JPEG-like block artifacts.
    Compare pixel differences on 8x8 boundaries vs non-boundary positions.
    """
    gray_f = gray.astype(np.float32)
    h, w = gray_f.shape[:2]

    if h < block_size * 2 or w < block_size * 2:
        return {
            "block_boundary_diff_mean": 0.0,
            "block_non_boundary_diff_mean": 0.0,
            "blockiness_ratio": 0.0,
        }

    diff_x = np.abs(gray_f[:, 1:] - gray_f[:, :-1])  # shape: h, w-1
    diff_y = np.abs(gray_f[1:, :] - gray_f[:-1, :])  # shape: h-1, w

    x_idx = np.arange(diff_x.shape[1]) + 1
    y_idx = np.arange(diff_y.shape[0]) + 1

    x_boundary_mask = (x_idx % block_size) == 0
    y_boundary_mask = (y_idx % block_size) == 0

    boundary_diffs = np.concatenate(
        [diff_x[:, x_boundary_mask].ravel(), diff_y[y_boundary_mask, :].ravel()]
    )
    non_boundary_diffs = np.concatenate(
        [diff_x[:, ~x_boundary_mask].ravel(), diff_y[~y_boundary_mask, :].ravel()]
    )

    boundary_mean = float(boundary_diffs.mean()) if boundary_diffs.size else 0.0
    non_boundary_mean = float(non_boundary_diffs.mean()) if non_boundary_diffs.size else 0.0
    ratio = boundary_mean / (non_boundary_mean + 1e-6)

    return {
        "block_boundary_diff_mean": round(boundary_mean, 3),
        "block_non_boundary_diff_mean": round(non_boundary_mean, 3),
        "blockiness_ratio": round(float(ratio), 3),
    }


def _add_reason(reasons: list, reason_details: list, code: str, message: str, value, threshold):
    reasons.append(code)
    reason_details.append(
        {
            "code": code,
            "message": message,
            "value": value,
            "threshold": threshold,
        }
    )


def check_image_quality(img_array: np.ndarray, config: dict = None) -> dict:
    """
    Quality Gate v1:
    - low_resolution
    - too_blurry
    - too_dark / too_bright
    - too_compressed (JPEG blockiness heuristic)
    """
    if config is None:
        config = {}

    thresholds = dict(_DEFAULT_THRESHOLDS)
    thresholds.update(config.get("quality_gate", {}))

    h, w = img_array.shape[:2]

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    reasons = []
    reason_details = []
    metrics = {}

    # 1) Resolution
    metrics["resolution"] = [w, h]
    if w < thresholds["min_width"] or h < thresholds["min_height"]:
        _add_reason(
            reasons,
            reason_details,
            "low_resolution",
            "Image resolution is below minimum requirement.",
            {"width": w, "height": h},
            {"min_width": thresholds["min_width"], "min_height": thresholds["min_height"]},
        )

    # 2) Blur (Laplacian variance)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["blur_score"] = round(blur_score, 2)
    if blur_score < thresholds["min_blur_score"]:
        _add_reason(
            reasons,
            reason_details,
            "too_blurry",
            "Image is too blurry for reliable inference.",
            round(blur_score, 2),
            {"min_blur_score": thresholds["min_blur_score"]},
        )

    # 3) Brightness
    brightness = float(np.mean(gray))
    metrics["brightness"] = round(brightness, 2)
    if brightness < thresholds["min_brightness"]:
        _add_reason(
            reasons,
            reason_details,
            "too_dark",
            "Image is too dark.",
            round(brightness, 2),
            {"min_brightness": thresholds["min_brightness"]},
        )
    elif brightness > thresholds["max_brightness"]:
        _add_reason(
            reasons,
            reason_details,
            "too_bright",
            "Image is too bright / overexposed.",
            round(brightness, 2),
            {"max_brightness": thresholds["max_brightness"]},
        )

    # 4) Heavy compression (blockiness heuristic)
    blockiness_metrics = _compute_blockiness(gray)
    metrics.update(blockiness_metrics)
    if thresholds.get("enable_compression_check", True):
        blockiness_ratio = blockiness_metrics["blockiness_ratio"]
        if blockiness_ratio > thresholds["max_blockiness_ratio"]:
            _add_reason(
                reasons,
                reason_details,
                "too_compressed",
                "Strong JPEG block artifacts detected.",
                blockiness_ratio,
                {"max_blockiness_ratio": thresholds["max_blockiness_ratio"]},
            )

    is_valid = len(reasons) == 0

    return {
        "is_valid": is_valid,
        "reasons": reasons,
        "reason_details": reason_details,
        "metrics": metrics,
        "applied_thresholds": thresholds,
    }
