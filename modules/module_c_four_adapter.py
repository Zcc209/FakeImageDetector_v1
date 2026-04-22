import os
from typing import Any


def run_four_fusion_placeholder(image_path: str, settings: dict) -> dict[str, Any]:
    """
    Adapter inspired by four.ipynb.
    - Removes hardcoded API keys from notebook.
    - Reads keys from env/settings.
    - Keeps a safe placeholder result when dependencies are unavailable.
    """
    gemini_key = settings.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    serp_key = settings.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")

    return {
        "four_adapter": True,
        "image_path": image_path,
        "gemini_key_loaded": bool(gemini_key),
        "serp_key_loaded": bool(serp_key),
        "note": "Install paddleocr/open_clip/google-generativeai and implement full fusion as needed.",
    }
