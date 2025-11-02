from __future__ import annotations

import os
import io
from typing import List, Optional
from PIL import Image

import fitz  # PyMuPDF


def extract_images_from_page(pdf_path: str, page_num: int) -> List[bytes]:
    """특정 페이지에서 모든 이미지를 추출하여 바이트 리스트로 반환"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return images
        
        page = doc[page_num - 1]  # 0-indexed
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                # 이미지 추출
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
            except Exception:
                continue
        
        doc.close()
    except Exception:
        pass
    
    return images


def get_first_image_from_page(pdf_path: str, page_num: int) -> Optional[bytes]:
    """특정 페이지에서 첫 번째 이미지만 추출"""
    images = extract_images_from_page(pdf_path, page_num)
    return images[0] if images else None

