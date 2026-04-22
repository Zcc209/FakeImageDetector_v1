import importlib.util
import os
import sys
from typing import Any

import numpy as np
from PIL import Image


class DeepfakeXceptionEngine:
    """
    DeepfakeBench Xception adapter from Deepfake_colab notebook.
    If model/repo/weights are missing, it returns a graceful error payload.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self._model = None
        self._transform = None
        self._device = None
        self._init_error: str | None = None

    def _init_model(self) -> None:
        if self._model is not None or self._init_error is not None:
            return
        if not self.settings.get("DEEPFAKE_XCEPTION_ENABLED", True):
            self._init_error = "DEEPFAKE_XCEPTION_ENABLED is false"
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms
            import types

            db_root = self.settings.get("DEEPFAKEBENCH_ROOT", "")
            ckpt_path = self.settings.get("DEEPFAKE_XCEPTION_CKPT", "")
            training_dir = os.path.join(db_root, "training")
            xception_py = os.path.join(training_dir, "networks", "xception.py")

            if not os.path.isfile(xception_py):
                raise FileNotFoundError(f"DeepfakeBench xception.py not found: {xception_py}")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Xception checkpoint not found: {ckpt_path}")

            if training_dir not in sys.path:
                sys.path.append(training_dir)

            # Some released xception.py files require metrics.registry.BACKBONE.
            # Provide a lightweight runtime shim so we can load the backbone
            # without importing full DeepfakeBench package tree.
            if "metrics.registry" not in sys.modules:
                metrics_mod = types.ModuleType("metrics")
                registry_mod = types.ModuleType("metrics.registry")

                class _DummyRegistry:
                    def register_module(self, module_name=None):
                        def _decorator(cls):
                            return cls

                        return _decorator

                registry_mod.BACKBONE = _DummyRegistry()
                metrics_mod.registry = registry_mod
                sys.modules["metrics"] = metrics_mod
                sys.modules["metrics.registry"] = registry_mod

            spec = importlib.util.spec_from_file_location("db_xception", xception_py)
            db_xception = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(db_xception)
            Xception = db_xception.Xception

            class XceptionDeepfakeWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = Xception(
                        {
                            "mode": "original",
                            "num_classes": 2,
                            "inc": 3,
                            "dropout": False,
                        }
                    )

                def forward(self, x):
                    feat = self.backbone.features(x)
                    cls = self.backbone.classifier(feat)
                    prob = torch.softmax(cls, dim=1)[:, 1]
                    return {"cls": cls, "prob": prob, "feat": feat}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = XceptionDeepfakeWrapper().to(device)

            ckpt = torch.load(ckpt_path, map_location=device)
            for key in ["state_dict", "model", "net", "weights"]:
                if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break

            normalized = {}
            for k, v in ckpt.items():
                nk = k[7:] if k.startswith("module.") else k
                normalized[nk] = v

            model.load_state_dict(normalized, strict=False)
            model.eval()

            self._transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            self._model = model
            self._device = device
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)

    @staticmethod
    def _crop_face(img_rgb: np.ndarray, box: list[int]) -> np.ndarray | None:
        if len(box) != 4:
            return None
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return img_rgb[y1:y2, x1:x2]

    def run(self, img_rgb: np.ndarray, scrfd_faces: list[dict[str, Any]]) -> dict[str, Any]:
        self._init_model()

        if not scrfd_faces:
            return {"deepfake_faces_count": 0, "deepfake_results": []}

        if self._model is None:
            return {
                "deepfake_faces_count": len(scrfd_faces),
                "deepfake_results": [],
                "deepfake_error": self._init_error or "Deepfake model unavailable",
            }

        import torch

        thr05 = float(self.settings.get("DEEPFAKE_FACE_THRESHOLD_05", 0.5))
        thr08 = float(self.settings.get("DEEPFAKE_FACE_THRESHOLD_08", 0.8))
        out_rows = []
        max_prob = 0.0

        with torch.no_grad():
            for idx, face in enumerate(scrfd_faces):
                box = face.get("box", [])
                crop = self._crop_face(img_rgb, box)
                if crop is None:
                    continue

                pil = Image.fromarray(crop).convert("RGB")
                x = self._transform(pil).unsqueeze(0).to(self._device)
                pred = self._model(x)
                fake_prob = float(pred["prob"].cpu().item())
                pred_idx = int(torch.argmax(pred["cls"], dim=1).cpu().item())
                label_05 = "fake" if pred_idx == 1 else "real"
                label_08 = "fake" if fake_prob >= thr08 else ("suspicious" if fake_prob >= thr05 else "real")
                max_prob = max(max_prob, fake_prob)

                out_rows.append(
                    {
                        "face_index": idx,
                        "box": box,
                        "fake_prob": round(fake_prob, 6),
                        "pred_label_0p5": label_05,
                        "pred_label_0p8": label_08,
                    }
                )

        return {
            "deepfake_faces_count": len(out_rows),
            "deepfake_results": out_rows,
            "deepfake_max_prob": round(float(max_prob), 6),
            "deepfake_any_fake_0p5": any(row["pred_label_0p5"] == "fake" for row in out_rows),
            "deepfake_any_fake_0p8": any(row["pred_label_0p8"] == "fake" for row in out_rows),
        }
