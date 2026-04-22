import numpy as np


def run_module_a(img_array: np.ndarray, settings: dict, vision_hint: dict | None = None) -> dict:
    # OCR/DIRE branch placeholder with route-aware hints.
    route = (vision_hint or {}).get("route_decision")
    out = {"ocr_text": "發現高風險文字"}
    if route == "no_person":
        out["ocr_branch_suggested"] = True
    if route == "person_without_face":
        out["openclip_branch_suggested"] = True
    return out
