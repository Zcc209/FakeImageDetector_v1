from pathlib import Path
import os
import json
from datetime import datetime

from config.settings import load_settings
from api.services.pipeline_service import run_pipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FakeImageDetector New Architecture CLI")
    parser.add_argument("source", type=str, help="Image local path or URL")
    parser.add_argument("--disable-fetcher", action="store_true", help="Disable URL fetch")
    args = parser.parse_args()

    settings = load_settings()
    result = run_pipeline(args.source, settings, disable_fetcher=args.disable_fetcher)

    os.makedirs(settings["OUTPUT_DIR"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(settings["OUTPUT_DIR"], f"result_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
