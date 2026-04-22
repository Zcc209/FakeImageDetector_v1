import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class TruForEngine:
    """TruFor adapter converted from Trufor_f.ipynb command flow."""

    def __init__(self, settings: dict):
        self.settings = settings

    def _resolve_test_py(self) -> str:
        test_py = self.settings.get("TRUFOR_TEST_PY", "test.py")
        if os.path.isabs(test_py):
            return test_py
        return os.path.join(self.settings.get("TRUFOR_WORK_DIR", ""), test_py)

    def _resolve_model_file(self) -> str:
        mf = self.settings.get("TRUFOR_MODEL_FILE", "pretrained_models/trufor.pth.tar")
        if os.path.isabs(mf):
            return mf
        return os.path.join(self.settings.get("TRUFOR_WORK_DIR", ""), mf)

    def _precheck(self) -> dict | None:
        micromamba_bin = self.settings.get("TRUFOR_MICROMAMBA_BIN", "micromamba")
        if os.path.isabs(micromamba_bin):
            mm_exists = os.path.exists(micromamba_bin)
        else:
            mm_exists = shutil.which(micromamba_bin) is not None

        env_prefix = self.settings.get("TRUFOR_ENV_PREFIX", "")
        test_py_abs = self._resolve_test_py()
        model_file_abs = self._resolve_model_file()
        work_dir = self.settings.get("TRUFOR_WORK_DIR", "")

        missing = []
        if not mm_exists:
            missing.append(f"micromamba not found: {micromamba_bin}")
        if env_prefix and (not os.path.isdir(env_prefix)):
            missing.append(f"trufor env prefix not found: {env_prefix}")
        if not os.path.isdir(work_dir):
            missing.append(f"work dir not found: {work_dir}")
        if not os.path.isfile(test_py_abs):
            missing.append(f"test.py not found: {test_py_abs}")
        if not os.path.isfile(model_file_abs):
            missing.append(f"model file not found: {model_file_abs}")

        if missing:
            return {
                "trufor_enabled": True,
                "trufor_score": None,
                "trufor_error": " ; ".join(missing),
            }
        return None

    def _build_cmd(self, img_path: Path, out_dir: Path) -> list[str]:
        env_prefix = self.settings.get("TRUFOR_ENV_PREFIX", "")
        cmd = [self.settings.get("TRUFOR_MICROMAMBA_BIN", "micromamba"), "run"]

        if env_prefix:
            cmd.extend(["-p", env_prefix])
        else:
            cmd.extend(["-n", self.settings.get("TRUFOR_ENV_NAME", "trufor")])

        cmd.extend(
            [
                "python",
                self.settings.get("TRUFOR_TEST_PY", "test.py"),
                "-g",
                str(self.settings.get("TRUFOR_GPU", 0)),
                "-in",
                str(img_path),
                "-out",
                str(out_dir),
                "-exp",
                self.settings.get("TRUFOR_EXP", "trufor_ph3"),
                "TEST.MODEL_FILE",
                self.settings.get("TRUFOR_MODEL_FILE", "pretrained_models/trufor.pth.tar"),
            ]
        )
        return cmd

    def run(self, img_array_rgb: np.ndarray) -> dict[str, Any]:
        if not self.settings.get("TRUFOR_ENABLED", False):
            return {
                "trufor_enabled": False,
                "trufor_score": None,
                "trufor_error": "TRUFOR_ENABLED is false",
            }

        pre = self._precheck()
        if pre:
            return pre

        work_dir = Path(self.settings.get("TRUFOR_WORK_DIR"))
        in_dir = Path(self.settings.get("TRUFOR_IN_DIR"))
        out_dir = Path(self.settings.get("TRUFOR_OUT_DIR"))
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"trufor_{ts}.jpg"
        img_path = in_dir / img_name
        npz_path = out_dir / f"{img_name}.npz"

        Image.fromarray(img_array_rgb).save(img_path)

        cmd = self._build_cmd(img_path, out_dir)

        env = os.environ.copy()
        mamba_root = self.settings.get("TRUFOR_MAMBA_ROOT_PREFIX")
        if mamba_root:
            env["MAMBA_ROOT_PREFIX"] = mamba_root

        try:
            cp = subprocess.run(
                cmd,
                cwd=str(work_dir),
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            return {
                "trufor_enabled": True,
                "trufor_score": None,
                "trufor_error": exc.stderr[-1200:] if exc.stderr else str(exc),
            }
        except Exception as exc:  # pragma: no cover
            return {
                "trufor_enabled": True,
                "trufor_score": None,
                "trufor_error": str(exc),
            }

        if not npz_path.exists():
            return {
                "trufor_enabled": True,
                "trufor_score": None,
                "trufor_error": f"NPZ output missing: {npz_path}",
                "trufor_stdout_tail": cp.stdout[-500:] if cp.stdout else "",
            }

        try:
            npz = np.load(npz_path)
            score = float(npz["score"])
            out = {
                "trufor_enabled": True,
                "trufor_score": round(score, 6),
                "trufor_npz": str(npz_path),
                "is_tampered": score > float(self.settings.get("TRUFOR_THRESHOLD", 0.5)),
            }
            if "map" in npz.files:
                out["trufor_map_shape"] = list(npz["map"].shape)
            if "conf" in npz.files:
                out["trufor_conf_shape"] = list(npz["conf"].shape)
            return out
        except Exception as exc:  # pragma: no cover
            return {
                "trufor_enabled": True,
                "trufor_score": None,
                "trufor_error": f"Failed to parse NPZ: {exc}",
            }
