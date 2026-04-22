import numpy as np

from modules.deepfake_engine import DeepfakeXceptionEngine
from modules.scrfd_engine import SCRFDEngine
from modules.trufor_engine import TruForEngine
from modules.yolo_engine import YOLOv8ONNXEngine


def _crop_from_box(img: np.ndarray, box: list[int]) -> np.ndarray | None:
    if len(box) != 4:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def run_module_b(img_array: np.ndarray, settings: dict) -> dict:
    """
    Vision branch (flowchart-aligned):
    - YOLOv8 object detection
    - SCRFD face detection
    - Branching: person+face -> Deepfake/TruFor, person+no-face -> OpenCLIP/SerpAPI path,
      no-person -> OCR/DIRE path
    """
    vision: dict = {}

    # 1) YOLOv8
    yolo = YOLOv8ONNXEngine(settings)
    vision.update(yolo.detect(img_array))

    person_name = settings.get("YOLO_PERSON_CLASS_NAME", "person")
    yolo_objects = vision.get("yolo_objects", [])
    person_objects = [o for o in yolo_objects if o.get("label") == person_name]
    has_person = len(person_objects) > 0

    # 2) SCRFD
    if has_person or settings.get("FORCE_SCRFD", True):
        scrfd = SCRFDEngine(settings)
        vision.update(scrfd.detect_faces(img_array))
    else:
        vision["scrfd_face_count"] = 0
        vision["scrfd_faces"] = []

    has_face = int(vision.get("scrfd_face_count", 0)) > 0

    # 3) route decision
    if has_person and has_face:
        route = "person_with_face"
        branch_targets = ["deepfake_check", "trufor"]
    elif has_person and not has_face:
        route = "person_without_face"
        branch_targets = ["openclip_serpapi"]
    else:
        route = "no_person"
        branch_targets = ["serpapi_dire_ocr"]

    vision["route_decision"] = route
    vision["branch_targets"] = branch_targets
    vision["has_person"] = has_person
    vision["has_face"] = has_face

    # 4) Build ROI hints from YOLO objects (for downstream processing)
    roi_crops = []
    for obj in person_objects:
        box = obj.get("box", [])
        crop = _crop_from_box(img_array, box)
        if crop is None:
            continue
        roi_crops.append(
            {
                "label": obj.get("label", "object"),
                "confidence": obj.get("confidence"),
                "box": box,
                "shape": list(crop.shape),
            }
        )
    vision["roi_crops"] = roi_crops

    # 5) Deepfake/TruFor branch for person+face
    if route == "person_with_face":
        deepfake = DeepfakeXceptionEngine(settings)
        vision.update(deepfake.run(img_array, vision.get("scrfd_faces", [])))

        trufor = TruForEngine(settings)
        vision.update(trufor.run(img_array))
    else:
        # Keep output schema stable for frontend.
        vision.setdefault("deepfake_faces_count", 0)
        vision.setdefault("deepfake_results", [])
        vision.setdefault("trufor_enabled", bool(settings.get("TRUFOR_ENABLED", False)))
        vision.setdefault("trufor_score", None)

    return vision
