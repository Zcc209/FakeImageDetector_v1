import os
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify

from config.settings import load_settings
from api.services.pipeline_service import run_pipeline

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.post("/analyze")
def analyze():
    settings = load_settings()

    if "file" in request.files:
        f = request.files["file"]
        if not f.filename:
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        os.makedirs(settings["INPUT_DIR"], exist_ok=True)
        filename = secure_filename(f.filename)
        temp_path = os.path.join(settings["INPUT_DIR"], filename)
        f.save(temp_path)
        source = temp_path
    else:
        payload = request.get_json(silent=True) or {}
        source = payload.get("source")

    if not source:
        return jsonify({"status": "error", "message": "Missing file or source"}), 400

    result = run_pipeline(source, settings)
    return jsonify(result), 200
