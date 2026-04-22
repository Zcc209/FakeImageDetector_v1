import numpy as np

from modules.scrfd_engine import SCRFDEngine
from modules.trufor_engine import TruForEngine


def _run_yolo_stub(img_array: np.ndarray, settings: dict) -> dict:
    """Temporary YOLO placeholder until real YOLO integration is added."""
    if settings.get("YOLO_STUB_ASSUME_PERSON", True):
        return {
            "yolo_objects": [
                {"label": "person", "confidence": 0.95, "box": [0, 0, 10, 10]}
            ]
        }
    return {"yolo_objects": []}


def run_module_b(img_array: np.ndarray, settings: dict) -> dict:
    """
    Vision branch:
    - YOLO (stub)
    - SCRFD (from SCRFD.ipynb)
    - TruFor (from Trufor_f.ipynb)
    """
    vision = {}
    vision.update(_run_yolo_stub(img_array, settings))

    # Route rule: run SCRFD if YOLO says person, or if force flag enabled.
    has_person = any(obj.get("label") == "person" for obj in vision.get("yolo_objects", []))
    if has_person or settings.get("FORCE_SCRFD", True):
        scrfd = SCRFDEngine(settings)
        vision.update(scrfd.detect_faces(img_array))
    else:
        vision["scrfd_face_count"] = 0
        vision["scrfd_faces"] = []

    # Optional TruFor branch
    trufor = TruForEngine(settings)
    vision.update(trufor.run(img_array))

    return vision
