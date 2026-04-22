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
        # Vision settings
        "YOLO_STUB_ASSUME_PERSON": True,
        "FORCE_SCRFD": True,
        "SCRFD_MODEL_ROOT": "/content/drive/MyDrive/FakeImageDetector/models_weights/scrfd" if is_colab else os.path.join(model_dir, "scrfd"),
        "SCRFD_MODEL_NAME": "buffalo_l",
        "SCRFD_DET_SIZE": (640, 640),
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
