import numpy as np

from modules.module_c_four_adapter import run_four_fusion_placeholder


def run_module_c(img_array: np.ndarray, settings: dict) -> dict:
    """
    Content/risk branch adapted from four.ipynb.
    Current implementation is a safe placeholder that keeps API-key handling clean.
    """
    out = {
        "risk_score": 0.3,
        "risk_level": "low",
        "explanation": "module_c placeholder",
    }

    if settings.get("ENABLE_FOUR_ADAPTER", False):
        out.update(run_four_fusion_placeholder("<in-memory-image>", settings))

    return out
