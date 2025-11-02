from __future__ import annotations

import os
from typing import Dict, Optional
from docx import Document


ROLE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "engine_department_roles.docx"
)


def _parse_role_docx() -> Dict[str, str]:
    """docx 파일을 파싱하여 직급별 정보를 딕셔너리로 반환"""
    if not os.path.exists(ROLE_FILE):
        return {}
    
    doc = Document(ROLE_FILE)
    roles: Dict[str, list] = {
        "3등 기관사": [],
        "2등 기관사": [],
        "1등 기관사": [],
        "기관장": [],
    }
    
    current_role: Optional[str] = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        # 직급 헤더 확인 - 더 정확한 패턴 매칭
        if text.startswith("1. 3등 기관사"):
            current_role = "3등 기관사"
            # 헤더도 포함
            roles[current_role].append(text)
            continue
        elif text.startswith("2. 2등 기관사"):
            current_role = "2등 기관사"
            roles[current_role].append(text)
            continue
        elif text.startswith("3. 1등 기관사"):
            current_role = "1등 기관사"
            roles[current_role].append(text)
            continue
        elif text.startswith("4. 기관장"):
            current_role = "기관장"
            roles[current_role].append(text)
            continue
        
        # 다른 섹션 시작 감지 (다음 직급으로 넘어감)
        if current_role and (
            text.startswith("1. ") or 
            text.startswith("2. ") or 
            text.startswith("3. ") or 
            text.startswith("4. ")
        ):
            # 숫자로 시작하는 새로운 섹션 발견 시 현재 직급 종료
            if "기관사" not in text and "기관장" not in text:
                current_role = None
        
        # 현재 직급에 텍스트 추가
        if current_role and current_role in roles:
            roles[current_role].append(text)
    
    # 리스트를 문자열로 변환
    return {
        role: "\n".join(content) 
        for role, content in roles.items()
        if content
    }


_role_cache: Optional[Dict[str, str]] = None


def get_role_info(role: str) -> str:
    """선택한 직급에 해당하는 정보를 반환"""
    global _role_cache
    if _role_cache is None:
        _role_cache = _parse_role_docx()
    
    return _role_cache.get(role, "")


def get_role_info_for_prompt(role: str) -> str:
    """프롬프트에 포함할 형식으로 직급 정보 반환"""
    info = get_role_info(role)
    if not info:
        return ""
    
    return f"\n[직급 정보: {role}]\n{info}\n"

