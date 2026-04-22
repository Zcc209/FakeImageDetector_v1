import os
from typing import Any

import cv2
import numpy as np


class SCRFDEngine:
    """SCRFD face detector adapter converted from SCRFD.ipynb."""

    def __init__(self, settings: dict):
        self.settings = settings
        self._app = None
        self._init_error = None

    def _init_model(self) -> None:
        if self._app is not None or self._init_error is not None:
            return

        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1

            model_root = self.settings.get(
                "SCRFD_MODEL_ROOT",
                os.path.join(self.settings["MODEL_DIR"], "scrfd"),
            )
            os.makedirs(model_root, exist_ok=True)

            app = FaceAnalysis(
                name=self.settings.get("SCRFD_MODEL_NAME", "buffalo_l"),
                root=model_root,
                allowed_modules=["detection"],
                providers=providers,
            )
            det_size = tuple(self.settings.get("SCRFD_DET_SIZE", (640, 640)))
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            self._app = app

        except Exception as exc:  # pragma: no cover - environment-dependent
            self._init_error = str(exc)

    def detect_faces(self, img_array_rgb: np.ndarray) -> dict[str, Any]:
        self._init_model()

        if self._app is None:
            return {
                "scrfd_face_count": 0,
                "scrfd_faces": [],
                "scrfd_error": self._init_error or "SCRFD model unavailable",
            }

        img_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        faces = self._app.get(img_bgr)

        out = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            score = float(getattr(face, "det_score", 0.0))
            out.append({"box": bbox, "score": round(score, 4)})

        return {
            "scrfd_face_count": len(out),
            "scrfd_faces": out,
        }


def draw_scrfd_overlay(img_array_rgb: np.ndarray, scrfd_faces: list[dict[str, Any]]) -> np.ndarray:
    """Draw green face boxes on an RGB image."""
    draw = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
    h, w = draw.shape[:2]
    thickness = max(2, int(min(h, w) / 300))
    font_scale = max(0.7, min(h, w) / 900)

    for face in scrfd_faces:
        box = face.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        score = face.get("score", 0.0)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            draw,
            f"face {score:.3f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
