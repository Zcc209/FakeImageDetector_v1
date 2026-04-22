import numpy as np
from PIL import Image, ImageOps

def preprocess_image(img: Image.Image, max_size: int = 1920) -> np.ndarray:
    """
    將圖片進行標準化前處理：EXIF轉正、轉為RGB格式、縮放尺寸，
    最後輸出 B 組與 C 組模型通用的 numpy.ndarray 格式。
    """
    
    # 1. 處理 EXIF 旋轉
    img = ImageOps.exif_transpose(img)
    
    # 2. 統一轉換為 RGB 格式 (升級版 Alpha 疊合邏輯)
    # 判斷圖片是否有透明通道或透明度資訊
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # 統一先轉為標準的 RGBA 格式
        img_rgba = img.convert("RGBA")
        # 建立一張完全不透明的純白底圖 (大小與原圖相同)
        background = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        # 使用 alpha_composite 完美的將原圖疊加在白底上
        alpha_composite = Image.alpha_composite(background, img_rgba)
        # 最後捨棄已經沒用的透明通道，轉換成純 RGB
        img = alpha_composite.convert("RGB")
    elif img.mode != "RGB":
        # 其他格式 (如純灰階 L) 直接轉 RGB
        img = img.convert("RGB")
            
    # 3. 限制最大尺寸
    width, height = img.size
    if max(width, height) > max_size:
        ratio = max_size / float(max(width, height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    # 4. 轉換為 numpy ndarray 格式
    img_array = np.array(img)
    
    return img_array