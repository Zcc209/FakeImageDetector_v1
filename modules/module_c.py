import numpy as np

from modules.module_c_four_adapter import run_four_fusion_placeholder


def run_module_c(img_array: np.ndarray, settings: dict) -> dict:
    """
    Content/risk branch adapted from four.ipynb.
    Keep safe placeholders while exposing branch readiness.
    """
    out = {
        "risk_score": 0.3,
        "risk_level": "low",
        "explanation": "module_c placeholder",
        "openclip_serpapi_ready": bool(settings.get("SERP_API_KEY")),
        "dire_ocr_ready": False,
    }

    if settings.get("ENABLE_FOUR_ADAPTER", False):
        out.update(run_four_fusion_placeholder("<in-memory-image>", settings))

    return out
