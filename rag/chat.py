from __future__ import annotations

from typing import List, Dict, Any, Optional
import heapq

from openai import OpenAI

from .embed import embed_query
from .index import load_index, search as faiss_search
from .store import list_manuals, load_chunks, manual_paths
from .role_parser import get_role_info_for_prompt
from .image_extractor import get_first_image_from_page, get_image_by_bbox


def _gather_candidates(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_vec = embed_query(query)

    candidates: List[Dict[str, Any]] = []
    manuals = list_manuals()

    for m in manuals:
        mid = m["id"]
        try:
            from .store import manual_paths
            idx = load_index(manual_paths(mid)["index"])
            scores, indices = faiss_search(idx, query_vec, top_k=top_k)
            chunks = load_chunks(mid)
            for s, i in zip(scores, indices):
                if i < 0 or i >= len(chunks):
                    continue
                c = chunks[i]
                candidates.append({
                    "manual_id": mid,
                    "score": float(s),
                    "chunk": c,
                })
        except Exception:
            continue

    # take global top_k by score (IP == cosine similarity)
    top = heapq.nlargest(top_k, candidates, key=lambda x: x["score"]) if candidates else []
    return top


def _build_context(cands: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts: List[str] = []
    for i, item in enumerate(cands, 1):
        ch = item["chunk"]
        header = ch.get("header", "")
        start_page = ch.get("start_page", "?")
        has_image = ch.get("has_image", False)
        content = (ch.get("content", "") or "")
        per_chunk = max_chars // max(1, len(cands))
        snippet_len = max(800, min(len(content), per_chunk))
        snippet = content[: snippet_len]
        image_note = " (이미지 포함)" if has_image else ""
        parts.append(
            f"--- 관련 문서 #{i} (제목: {header}, 페이지: {start_page}{image_note}) ---\n{snippet}\n"
        )
    return "\n".join(parts)


def _get_language_instruction(language: str) -> str:
    """언어에 따른 답변 지시문 반환"""
    lang_map = {
        "한국어": "한국어로",
        "영어": "영어로 (in English)",
        "중국어": "중국어로 (用中文)",
        "일본어": "일본어로 (日本語で)"
    }
    return lang_map.get(language, "한국어로")


def _build_prompt(context: str, query: str, language: str = "한국어", role: Optional[str] = None) -> str:
    lang_instruction = _get_language_instruction(language)
    role_info = get_role_info_for_prompt(role) if role else ""
    
    prompt_parts = [
        f"당신은 선박 기기 매뉴얼 전문가입니다. 아래 근거 문서만 활용해 간결/정확한 {lang_instruction} 답변을 작성하세요. 모호하면 추가 질문을 요청하세요."
    ]
    
    if role_info:
        prompt_parts.append(role_info)
        prompt_parts.append(f"답변 시 위 직급 정보를 고려하여 해당 직급에 적합한 수준과 내용으로 설명하세요.")
    
    prompt_parts.extend([
        "\n[근거]\n" + context + "\n\n[질문]\n" + query + "\n\n",
        "요구: 문서에서 해당되는 모든 사례/항목을 가능한 빠짐없이 번호로 나열하고, 각 항목에 대해 핵심 Bullet→절차→주의/한계를 포함하세요. 마지막에 참고 출처(제목·페이지)."
    ])
    
    return "\n".join(prompt_parts)


def answer(query: str, top_k: int = 5, language: str = "한국어", role: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    # Guard: no manuals
    manuals = list_manuals()
    if not manuals:
        return {
            "answer": "업로드된 매뉴얼이 없습니다. 상단 '소스 업로드'로 PDF를 등록·인덱싱한 뒤 다시 질문해 주세요.",
            "citations": [],
        }

    cands = _gather_candidates(query, top_k=top_k)
    if not cands:
        return {
            "answer": "관련 문서를 찾지 못했습니다. 매뉴얼 업로드/인덱싱 상태를 확인하거나 질문을 더 구체화해 주세요.",
            "citations": [],
        }

    context = _build_context(cands)
    prompt = _build_prompt(context, query, language=language, role=role)

    # 대화 히스토리 구성
    messages = [
        {"role": "system", "content": "당신은 정확한 기술 매뉴얼 어시스턴트입니다. 이전 대화 맥락을 고려하여 답변하되, 항상 제공된 근거 문서를 기반으로 답변하세요."}
    ]
    
    # 이전 대화 히스토리 추가 (최근 5턴만 유지)
    if conversation_history:
        # 최근 5턴만 사용 (너무 길어지면 토큰 초과)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": prompt})

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
    )
    text = resp.choices[0].message.content

    citations = []
    images_data = []
    
    for it in cands:
        chunk = it["chunk"]
        manual_id = it["manual_id"]
        has_image = chunk.get("has_image", False)
        page_num = chunk.get("start_page")
        
        citation = {
            "title": chunk.get("header", ""),
            "page": page_num if page_num != "?" else "?",
            "score": it["score"],
            "has_image": has_image,
        }
        citations.append(citation)
        
        # 이미지가 있는 경우 이미지 추출
        if has_image and page_num and page_num != "?":
            try:
                pdf_path = manual_paths(manual_id)["pdf"]
                # bbox 정보가 있으면 특정 위치의 이미지 추출, 없으면 첫 번째 이미지 추출
                image_bbox = chunk.get("image_bbox")
                if image_bbox:
                    image_bytes = get_image_by_bbox(pdf_path, int(page_num), image_bbox)
                else:
                    image_bytes = get_first_image_from_page(pdf_path, int(page_num))
                
                if image_bytes:
                    images_data.append({
                        "title": chunk.get("header", ""),
                        "page": page_num,
                        "image_bytes": image_bytes,
                    })
            except Exception:
                pass
    
    return {"answer": text, "citations": citations, "images": images_data}
