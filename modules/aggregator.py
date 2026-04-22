def aggregate_result(source: str, img_array, gate_result: dict, routed: dict) -> dict:
    return {
        "status": "success",
        "metadata": {
            "source": source,
            "processed_shape": list(img_array.shape),
        },
        "gate_details": gate_result,
        "vision": routed.get("module_b", {}),
        "content": {
            **routed.get("module_a", {}),
            **routed.get("module_c", {}),
        },
    }
