from utils.image_io import load_image
from utils.preprocess import preprocess_image
from utils.quality_gate import check_image_quality
from utils.enhance import enhance_by_quality_reasons
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
    max_retry = int(settings.get("QUALITY_RETRY_MAX", 2))
    enhancement_history = []
    gate = check_image_quality(img_array, gate_input)

    for _ in range(max_retry):
        if gate["is_valid"]:
            break

        enhanced, actions = enhance_by_quality_reasons(img_array, gate.get("reasons", []))
        if not actions:
            break

        img_array = enhanced
        enhancement_history.append({
            "actions": actions,
            "after_shape": list(img_array.shape),
        })
        gate = check_image_quality(img_array, gate_input)

    gate["enhancement_history"] = enhancement_history
    gate["enhancement_attempts"] = len(enhancement_history)

    if not gate["is_valid"]:
        return {
            "status": "rejected",
            "metadata": {"source": source},
            "gate_details": gate,
        }

    routed = route_modules(img_array, settings)
    return aggregate_result(source, img_array, gate, routed)
