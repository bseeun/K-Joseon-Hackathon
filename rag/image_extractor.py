from __future__ import annotations

import os
import io
from typing import List, Optional, Dict, Any
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


def get_image_by_bbox(pdf_path: str, page_num: int, bbox: Dict[str, float], tolerance: float = 5.0) -> Optional[bytes]:
    """
    특정 페이지에서 bbox 좌표와 일치하는 이미지를 추출
    
    Args:
        pdf_path: PDF 파일 경로
        page_num: 페이지 번호 (1부터 시작)
        bbox: 이미지의 bbox 정보 {"x0": float, "y0": float, "x1": float, "y1": float}
        tolerance: 좌표 일치 허용 오차 (기본 5.0 픽셀)
    
    Returns:
        이미지 바이트 데이터 또는 None
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        
        page = doc[page_num - 1]  # 0-indexed
        image_list = page.get_images(full=True)
        
        # bbox를 fitz.Rect로 변환
        target_bbox = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
        
        for img_info in image_list:
            try:
                # 이미지의 bbox 가져오기
                img_bbox = page.get_image_bbox(img_info)
                
                # bbox가 일치하는지 확인 (tolerance 범위 내)
                if (
                    abs(img_bbox.x0 - target_bbox.x0) <= tolerance and
                    abs(img_bbox.y0 - target_bbox.y0) <= tolerance and
                    abs(img_bbox.x1 - target_bbox.x1) <= tolerance and
                    abs(img_bbox.y1 - target_bbox.y1) <= tolerance
                ):
                    # 일치하는 이미지 추출
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    doc.close()
                    return image_bytes
            except Exception:
                continue
        
        doc.close()
    except Exception:
        pass
    
    return None

