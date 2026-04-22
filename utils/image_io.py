import os
import requests
import cv2
import numpy as np
import io
from .errors import AppError, ErrorCode
from io import BytesIO
from PIL import Image, UnidentifiedImageError

class ImageLoadError(Exception):
    """自定義的圖片讀取錯誤，方便上層系統捕捉"""
    pass

def load_image(source: str) -> Image.Image:
    """
    統一讀取圖片的入口函數。
    根據輸入字串自動判斷是 URL 還是本地路徑。
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _download_image(source)
    else:
        return _load_local_image(source)

def _download_image(url: str, timeout: int = 10) -> Image.Image:
    """從 URL 下載圖片並轉換為 PIL Image"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content))
        img.load() 
        return img
    except requests.exceptions.RequestException as e:
        raise ImageLoadError(f"網路圖片下載失敗: {e}")
    except UnidentifiedImageError:
        raise ImageLoadError(f"無法辨識的圖片格式: {url}")
    except Exception as e:
        raise ImageLoadError(f"未期的錯誤: {e}")

def _load_local_image(path: str) -> Image.Image:
    """從本地端讀取圖片並轉換為 PIL Image"""
    if not os.path.exists(path):
        raise ImageLoadError(f"找不到檔案: {path}")
    try:
        img = Image.open(path)
        img.load() 
        return img
    except UnidentifiedImageError:
        raise ImageLoadError(f"無法辨識的圖片格式: {path}")
    except Exception as e:
        raise ImageLoadError(f"未期的錯誤: {e}")

def load_image_from_bytes(image_bytes: bytes):
    """
    從 bytes 讀取圖片並轉為 PIL Image 物件，
    以配合 preprocess_image 中的 EXIF 旋轉處理。
    """
    try:
        # 使用 io.BytesIO 將 bytes 轉換為檔案流，再交給 PIL 讀取
        img = Image.open(io.BytesIO(image_bytes))
        
        # 呼叫 load() 強制載入圖片資料，若圖片破檔會在這裡直接噴錯攔截
        img.load() 
        
        # 統一轉換為 RGB 模式 (避免 RGBA 或是黑白圖片造成後續維度錯誤)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return img
        
    except Exception as e:
        raise AppError(ErrorCode.INVALID_INPUT, f"無法解析圖片資料，檔案可能已損毀或格式不支援: {str(e)}")