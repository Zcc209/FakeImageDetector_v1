from modules.module_a import run_module_a
from modules.module_b import run_module_b
from modules.module_c import run_module_c


def route_modules(img_array, settings: dict) -> dict:
    content_a = run_module_a(img_array, settings)
    vision_b = run_module_b(img_array, settings)
    fusion_c = run_module_c(img_array, settings)

    return {
        "module_a": content_a,
        "module_b": vision_b,
        "module_c": fusion_c,
    }
