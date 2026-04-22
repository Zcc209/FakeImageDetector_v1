from utils.image_io import load_image
from utils.preprocess import preprocess_image
from utils.quality_gate import check_image_quality
from modules.router import route_modules
from modules.aggregator import aggregate_result


def run_pipeline(source: str, settings: dict, disable_fetcher: bool = False) -> dict:
    is_url = source.startswith("http://") or source.startswith("https://")
    if is_url and disable_fetcher:
        return {
            "status": "error",
            "message": "URL input is disabled.",
            "metadata": {"source": source},
        }

    raw = load_image(source)
    img_array = preprocess_image(raw, max_size=settings["MAX_IMAGE_SIZE"])

    gate_input = {"quality_gate": settings.get("QUALITY_GATE", {})}
    gate = check_image_quality(img_array, gate_input)
    if not gate["is_valid"]:
        return {
            "status": "rejected",
            "metadata": {"source": source},
            "gate_details": gate,
        }

    routed = route_modules(img_array, settings)
    return aggregate_result(source, img_array, gate, routed)
