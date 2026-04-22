# src/core/errors.py
from enum import Enum

class ErrorCode(Enum):
    SUCCESS = 0
    # 系統與輸入層級錯誤 (100-199)
    UNKNOWN_ERROR = 100
    TIMEOUT_ERROR = 101
    FILE_NOT_FOUND = 102
    INVALID_INPUT = 103      # 非圖片、破檔等
    
    # 視覺模組層級錯誤 (200-299)
    MODEL_OOM = 200          # 記憶體爆滿
    VISION_INFERENCE_FAILED = 201

    # 內容與外部 API 層級錯誤 (300-399)
    API_RATE_LIMIT = 300
    EXTERNAL_API_FAILED = 301

class AppError(Exception):
    """自訂的系統錯誤類別，可用於主動拋出已知錯誤"""
    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

def build_error_response(code: ErrorCode, message: str) -> dict:
    """產生標準化的錯誤回傳 JSON 格式"""
    return {
        "status": "error",
        "error_code": code.value,
        "error_name": code.name,
        "message": message,
        "data": {} # 發生錯誤時，data 留空或放降級資訊
    }

def build_success_response(data: dict) -> dict:
    """產生標準化的成功回傳 JSON 格式"""
    return {
        "status": "success",
        "error_code": ErrorCode.SUCCESS.value,
        "error_name": ErrorCode.SUCCESS.name,
        "message": "OK",
        "data": data
    }