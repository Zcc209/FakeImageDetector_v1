import os

# Direct code switch: set True/False here.
TRUFOR_ENABLED_DEFAULT = True


def _is_colab() -> bool:
    return os.path.exists("/content") and "COLAB_RELEASE_TAG" in os.environ


def _first_existing(paths: list[str], fallback: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return fallback


def load_settings() -> dict:
    is_colab = _is_colab()
    base_dir = "/content/FakeImageDetector_NewArch" if is_colab else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(base_dir, "models")

    # Colab paths
    trufor_colab_base = os.getenv("TRUFOR_COLAB_BASE", "/content/drive/MyDrive/TruFor_colab")
    trufor_colab_repo = os.getenv("TRUFOR_REPO_DIR", "/content/TruFor/TruFor_train_test")

    # Local paths
    trufor_local_root = os.path.join(model_dir, "trufor")
    trufor_local_repo = os.getenv("TRUFOR_LOCAL_REPO_DIR", os.path.join(trufor_local_root, "TruFor_train_test"))

    trufor_model_candidates = [
        os.getenv("TRUFOR_LOCAL_MODEL_FILE", ""),
        os.path.join(trufor_local_root, "pretrained_models", "trufor.pth.tar"),
        os.path.join(trufor_local_root, "weights", "trufor.pth.tar"),
        os.path.join(trufor_local_repo, "pretrained_models", "trufor.pth.tar"),
    ]
    trufor_local_model = _first_existing(trufor_model_candidates, trufor_model_candidates[1])

    micromamba_candidates = [
        os.getenv("TRUFOR_MICROMAMBA_BIN", ""),
        r"C:\Users\linzi\AppData\Local\micromamba\micromamba.exe",  # preferred local binary
        r"C:\Users\linzi\micromamba\micromamba.exe",  # fallback if user installs here
    ]
    micromamba_local = _first_existing(micromamba_candidates, "micromamba")

    # Use your actual micromamba root by default (contains envs/trufor)
    mamba_root_local = os.getenv("TRUFOR_MAMBA_ROOT_PREFIX", r"C:\Users\linzi\AppData\Local\micromamba")
    mamba_root_colab = os.getenv("TRUFOR_MAMBA_ROOT_PREFIX", "/content/micromamba")
    trufor_env_prefix_local = os.getenv("TRUFOR_ENV_PREFIX", r"C:\Users\linzi\AppData\Local\micromamba\envs\trufor")
    yolo_local_dir = os.path.join(model_dir, "yolo")
    deepfakebench_root_local = os.getenv("DEEPFAKEBENCH_ROOT", os.path.join(model_dir, "deepfakebench"))
    deepfake_ckpt_local = os.getenv(
        "DEEPFAKE_XCEPTION_CKPT",
        os.path.join(deepfakebench_root_local, "xception_best.pth"),
    )

    settings = {
        "BASE_DIR": base_dir,
        "INPUT_DIR": os.path.join(base_dir, "input"),
        "OUTPUT_DIR": os.path.join(base_dir, "outputs"),
        "MODEL_DIR": model_dir,
        "CONF_THRESHOLD": 0.25,
        "MAX_IMAGE_SIZE": 1920,
        "QUALITY_GATE": {
            "min_width": 150,
            "min_height": 150,
            "min_blur_score": 80.0,
            "min_brightness": 30.0,
            "max_brightness": 240.0,
            "max_blockiness_ratio": 1.8,
            "enable_compression_check": True,
        },
        "QUALITY_RETRY_MAX": int(os.getenv("QUALITY_RETRY_MAX", "2")),
        # Vision settings
        "YOLO_ENABLED": os.getenv("YOLO_ENABLED", "true").lower() == "true",
        "YOLO_MODEL_PATH": os.getenv("YOLO_MODEL_PATH", os.path.join(yolo_local_dir, "yolov8n.onnx")),
        "YOLO_CLASSES_PATH": os.getenv("YOLO_CLASSES_PATH", os.path.join(yolo_local_dir, "classes.txt")),
        "YOLO_INPUT_SIZE": int(os.getenv("YOLO_INPUT_SIZE", "640")),
        "YOLO_CONF_THRESHOLD": float(os.getenv("YOLO_CONF_THRESHOLD", "0.25")),
        "YOLO_IOU_THRESHOLD": float(os.getenv("YOLO_IOU_THRESHOLD", "0.45")),
        "YOLO_MAX_DET": int(os.getenv("YOLO_MAX_DET", "100")),
        "YOLO_PERSON_CLASS_NAME": os.getenv("YOLO_PERSON_CLASS_NAME", "person"),
        "FORCE_SCRFD": True,
        "SCRFD_MODEL_ROOT": "/content/drive/MyDrive/FakeImageDetector/models_weights/scrfd" if is_colab else os.path.join(model_dir, "scrfd"),
        "SCRFD_MODEL_NAME": "buffalo_l",
        "SCRFD_DET_SIZE": (640, 640),
        # DeepfakeBench Xception (from Deepfake_colab notebook)
        "DEEPFAKE_XCEPTION_ENABLED": os.getenv("DEEPFAKE_XCEPTION_ENABLED", "true").lower() == "true",
        "DEEPFAKEBENCH_ROOT": os.getenv("DEEPFAKEBENCH_ROOT", "/content/DeepfakeBench" if is_colab else deepfakebench_root_local),
        "DEEPFAKE_XCEPTION_CKPT": os.getenv(
            "DEEPFAKE_XCEPTION_CKPT",
            "/content/drive/MyDrive/SCRFD_Colab/models/deepfakebench/xception_best.pth" if is_colab else deepfake_ckpt_local,
        ),
        "DEEPFAKE_FACE_THRESHOLD_05": float(os.getenv("DEEPFAKE_FACE_THRESHOLD_05", "0.5")),
        "DEEPFAKE_FACE_THRESHOLD_08": float(os.getenv("DEEPFAKE_FACE_THRESHOLD_08", "0.8")),
        # TruFor settings
        "TRUFOR_ENABLED": TRUFOR_ENABLED_DEFAULT,
        "TRUFOR_THRESHOLD": 0.5,
        "TRUFOR_MICROMAMBA_BIN": "/content/bin/micromamba" if is_colab else micromamba_local,
        "TRUFOR_MAMBA_ROOT_PREFIX": mamba_root_colab if is_colab else mamba_root_local,
        "TRUFOR_ENV_NAME": os.getenv("TRUFOR_ENV_NAME", "trufor"),
        "TRUFOR_ENV_PREFIX": "" if is_colab else trufor_env_prefix_local,
        "TRUFOR_TEST_PY": os.getenv("TRUFOR_TEST_PY", "test.py"),
        "TRUFOR_EXP": os.getenv("TRUFOR_EXP", "trufor_ph3"),
        "TRUFOR_GPU": int(os.getenv("TRUFOR_GPU", "0")),
        "TRUFOR_WORK_DIR": trufor_colab_repo if is_colab else trufor_local_repo,
        "TRUFOR_IN_DIR": os.path.join(trufor_colab_base, "images") if is_colab else os.path.join(base_dir, "runs", "trufor_in"),
        "TRUFOR_OUT_DIR": os.path.join(trufor_colab_base, "out") if is_colab else os.path.join(base_dir, "runs", "trufor_out"),
        "TRUFOR_MODEL_FILE": "pretrained_models/trufor.pth.tar" if is_colab else trufor_local_model,
        # four.ipynb adapter settings
        "ENABLE_FOUR_ADAPTER": False,
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        "SERP_API_KEY": os.getenv("SERP_API_KEY", ""),
    }

    # Optional runtime override for switch
    if os.getenv("TRUFOR_ENABLED") is not None:
        settings["TRUFOR_ENABLED"] = os.getenv("TRUFOR_ENABLED", "false").lower() == "true"

    return settings
