# New Architecture Project

This folder was generated from the previous `FakeImageDetector` codebase and remapped to the new architecture in `新版專題架構.md`.

## Run API (local)

```bash
python api/app.py
```

## Run CLI (local)

```bash
python main.py "input/test.jpg"
```

## Frontend

Open `http://localhost:8000` after backend starts.

## Colab runtime notes (SCRFD + TruFor)

`config/settings.py` auto-detects Colab and will use these defaults:
- `SCRFD_MODEL_ROOT=/content/drive/MyDrive/FakeImageDetector/models_weights/scrfd`
- `TRUFOR_WORK_DIR=/content/TruFor/TruFor_train_test`
- `TRUFOR_IN_DIR=/content/drive/MyDrive/TruFor_colab/images`
- `TRUFOR_OUT_DIR=/content/drive/MyDrive/TruFor_colab/out`
- `TRUFOR_MICROMAMBA_BIN=/content/bin/micromamba`
- `TRUFOR_ENABLED=True` (in Colab)

Optional overrides in Colab:

```python
import os
os.environ["TRUFOR_COLAB_BASE"] = "/content/drive/MyDrive/TruFor_colab"
os.environ["TRUFOR_REPO_DIR"] = "/content/TruFor/TruFor_train_test"
os.environ["TRUFOR_ENABLED"] = "true"
os.environ["TRUFOR_GPU"] = "0"
```
