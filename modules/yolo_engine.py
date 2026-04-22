import os
from typing import Any

import cv2
import numpy as np


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area2 = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union = area1 + area2 - inter + 1e-6
    return inter / union


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= max_det or order.size == 1:
            break
        ious = _compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return keep


class YOLOv8ONNXEngine:
    """YOLOv8 ONNX detector (Ultralytics export format)."""

    def __init__(self, settings: dict):
        self.settings = settings
        self._session = None
        self._classes: list[str] = []
        self._init_error: str | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None

    def _load_classes(self) -> None:
        path = self.settings.get("YOLO_CLASSES_PATH", "")
        if not path or not os.path.isfile(path):
            self._classes = []
            return
        with open(path, "r", encoding="utf-8") as f:
            self._classes = [line.strip() for line in f if line.strip()]

    def _init(self) -> None:
        if self._session is not None or self._init_error is not None:
            return
        try:
            import onnxruntime as ort

            model_path = self.settings.get("YOLO_MODEL_PATH", "")
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(f"YOLO model not found: {model_path}")

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._load_classes()
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)

    def _preprocess(self, img_rgb: np.ndarray, input_size: int) -> tuple[np.ndarray, float, int, int]:
        h, w = img_rgb.shape[:2]
        scale = min(input_size / max(w, 1), input_size / max(h, 1))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        dw = (input_size - nw) // 2
        dh = (input_size - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return blob, scale, dw, dh

    def detect(self, img_rgb: np.ndarray) -> dict[str, Any]:
        self._init()
        if self._session is None:
            return {
                "yolo_objects": [],
                "yolo_error": self._init_error or "YOLO init failed",
                "yolo_enabled": bool(self.settings.get("YOLO_ENABLED", True)),
            }

        if not self.settings.get("YOLO_ENABLED", True):
            return {"yolo_objects": [], "yolo_enabled": False}

        input_size = int(self.settings.get("YOLO_INPUT_SIZE", 640))
        conf_thr = float(self.settings.get("YOLO_CONF_THRESHOLD", 0.25))
        iou_thr = float(self.settings.get("YOLO_IOU_THRESHOLD", 0.45))
        max_det = int(self.settings.get("YOLO_MAX_DET", 100))

        blob, scale, dw, dh = self._preprocess(img_rgb, input_size)
        pred = self._session.run([self._output_name], {self._input_name: blob})[0]

        # Common YOLOv8 ONNX output: [1, 84, anchors].
        pred = np.squeeze(pred, axis=0)
        if pred.ndim != 2:
            return {"yolo_objects": [], "yolo_error": f"Unexpected output shape: {list(pred.shape)}"}
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # [anchors, 84]

        if pred.shape[1] < 6:
            return {"yolo_objects": [], "yolo_error": f"Unexpected output channels: {pred.shape[1]}"}

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        keep = scores >= conf_thr
        if not np.any(keep):
            return {"yolo_objects": [], "yolo_enabled": True}

        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        scores = scores[keep]

        # xywh -> xyxy (letterboxed coordinates)
        xyxy = np.zeros_like(boxes_xywh, dtype=np.float32)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # undo letterbox
        h, w = img_rgb.shape[:2]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / max(scale, 1e-6)
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / max(scale, 1e-6)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

        selected = _nms_xyxy(xyxy, scores, iou_thr=iou_thr, max_det=max_det)
        objects: list[dict[str, Any]] = []
        for i in selected:
            cls_id = int(cls_ids[i])
            label = self._classes[cls_id] if 0 <= cls_id < len(self._classes) else f"class_{cls_id}"
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            objects.append(
                {
                    "label": label,
                    "class_id": cls_id,
                    "confidence": round(float(scores[i]), 4),
                    "box": [x1, y1, x2, y2],
                }
            )

        person_name = self.settings.get("YOLO_PERSON_CLASS_NAME", "person")
        has_person = any(o.get("label") == person_name for o in objects)
        return {
            "yolo_enabled": True,
            "yolo_objects": objects,
            "yolo_has_person": has_person,
            "yolo_meta": {
                "model_path": self.settings.get("YOLO_MODEL_PATH", ""),
                "classes_count": len(self._classes),
                "conf_threshold": conf_thr,
                "iou_threshold": iou_thr,
            },
        }
