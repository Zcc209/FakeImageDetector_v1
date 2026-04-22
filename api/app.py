import os

from flask import Flask, send_from_directory
from flask_cors import CORS

from api.routes.analyze import analyze_bp


def create_app() -> Flask:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    frontend_dir = os.path.join(base_dir, "frontend")

    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    CORS(app)
    app.register_blueprint(analyze_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}, 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
