from __future__ import annotations

from typing import List, Dict, Any, Optional
import random

from openai import OpenAI

from .store import load_chunks
from .role_parser import get_role_info_for_prompt


def _sample_context(chunks: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    random.seed(42)
    random.shuffle(chunks)
    acc = []
    total = 0
    for ch in chunks:
        piece = f"제목: {ch.get('header','')}\n내용: {ch.get('content','')}\n"
        if total + len(piece) > max_chars:
            break
        acc.append(piece)
        total += len(piece)
    return "\n".join(acc)


def _get_language_instruction(language: str) -> str:
    """언어에 따른 퀴즈 생성 지시문 반환"""
    lang_map = {
        "한국어": "한국어로",
        "영어": "영어로 (in English)",
        "중국어": "중국어로 (用中文)",
        "일본어": "일본어로 (日本語で)"
    }
    return lang_map.get(language, "한국어로")


def generate_quiz(
    manual_id: str, 
    num_questions: int = 5, 
    language: str = "한국어",
    role: Optional[str] = None
) -> List[Dict[str, Any]]:
    chunks = load_chunks(manual_id)
    ctx = _sample_context(chunks)
    lang_instruction = _get_language_instruction(language)
    role_info = get_role_info_for_prompt(role) if role else ""

    prompt_parts = [
        f"아래 선박 매뉴얼 내용을 바탕으로 객관식 퀴즈를 {lang_instruction} 만들어주세요."
    ]
    
    if role_info:
        prompt_parts.append(role_info)
        prompt_parts.append(f"퀴즈 문제는 위 직급 정보를 고려하여 해당 직급에 적합한 수준의 난이도와 내용으로 출제하세요.")
    
    prompt_parts.extend([
        "출력 형식은 JSON 배열이며 각 원소는 다음 키를 가집니다: \n",
        "{question: str, options: [str,str,str,str], answer_index: int, citation: {title: str, page: int}}\n",
        f"[자료]\n{ctx}\n",
        f"문항 수: {num_questions}\n"
    ])
    
    prompt = "\n".join(prompt_parts)

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 정확한 시험 문제 출제자입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or "[]"

    # Try to parse JSON; if fails, fallback to LLM-based generation per chunk
    import json
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            return data[:num_questions]
    except Exception:
        pass

    # Fallback: 각 청크별로 LLM을 사용해서 퀴즈 생성
    qs: List[Dict[str, Any]] = []
    random.shuffle(chunks)  # 질문 순서를 다양하게
    
    for i in range(min(num_questions, len(chunks))):
        ch = chunks[i]
        header = ch.get('header', '')
        content = str(ch.get('content', '')).strip()
        page = ch.get("start_page", 0)
        
        if not content:
            continue
        
        # 각 청크에 대해 개별적으로 LLM으로 퀴즈 생성
        chunk_prompt_parts = [
            f"아래 선박 매뉴얼 내용을 바탕으로 객관식 퀴즈를 {lang_instruction} 만들어주세요."
        ]
        
        if role_info:
            chunk_prompt_parts.append(role_info)
            chunk_prompt_parts.append(f"퀴즈 문제는 위 직급 정보를 고려하여 해당 직급에 적합한 수준의 난이도와 내용으로 출제하세요.")
        
        chunk_prompt_parts.extend([
            "출력 형식은 JSON 객체이며 다음 키를 가집니다: \n",
            "{question: str, options: [str,str,str,str], answer_index: int}\n",
            f"[자료]\n제목: {header}\n내용: {content[:500]}\n",
            "하나의 객관식 문제만 생성하세요. answer_index는 0부터 3 사이의 정수입니다."
        ])
        
        chunk_prompt = "\n".join(chunk_prompt_parts)
        
        # LLM으로 퀴즈 생성 시도 (최대 2번 시도)
        success = False
        for attempt in range(2):
            try:
                chunk_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 정확한 시험 문제 출제자입니다. 반드시 JSON 형식으로만 응답하세요."},
                        {"role": "user", "content": chunk_prompt if attempt == 0 else f"{chunk_prompt}\n\n중요: 반드시 유효한 JSON 형식으로만 응답하세요. 예시: {{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"answer_index\": 0}}"},
                    ],
                    temperature=0.2,
                )
                chunk_text = chunk_resp.choices[0].message.content or "{}"
                
                # JSON 파싱 시도
                try:
                    # JSON 코드 블록 제거 시도
                    import re
                    json_match = re.search(r'\{[^{}]*\}', chunk_text, re.DOTALL)
                    if json_match:
                        chunk_text = json_match.group(0)
                    
                    chunk_data = json.loads(chunk_text)
                    if isinstance(chunk_data, dict) and "question" in chunk_data and "options" in chunk_data:
                        # options가 4개인지 확인
                        if isinstance(chunk_data["options"], list) and len(chunk_data["options"]) == 4:
                            # answer_index 유효성 확인
                            ans_idx = chunk_data.get("answer_index", 0)
                            if isinstance(ans_idx, int) and 0 <= ans_idx < 4:
                                chunk_data["citation"] = {"title": header, "page": page}
                                qs.append(chunk_data)
                                success = True
                                break
                except Exception:
                    continue
            except Exception:
                continue
        
        # LLM 호출이 모두 실패한 경우, 더 간단한 프롬프트로 최종 시도
        if not success:
            try:
                simple_prompt = (
                    f"아래 내용으로 객관식 문제 하나를 {lang_instruction} 만들어주세요.\n"
                    f"제목: {header}\n"
                    f"내용: {content[:300]}\n\n"
                    "JSON 형식으로 응답: {{\"question\": \"질문\", \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"], \"answer_index\": 0}}"
                )
                
                final_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "JSON 형식으로만 응답하세요."},
                        {"role": "user", "content": simple_prompt},
                    ],
                    temperature=0.2,
                )
                final_text = final_resp.choices[0].message.content or "{}"
                
                # JSON 추출 및 파싱
                import re
                json_match = re.search(r'\{[^{}]*\}', final_text, re.DOTALL)
                if json_match:
                    final_text = json_match.group(0)
                
                final_data = json.loads(final_text)
                if isinstance(final_data, dict) and "question" in final_data and "options" in final_data:
                    if isinstance(final_data["options"], list) and len(final_data["options"]) == 4:
                        final_data["citation"] = {"title": header, "page": page}
                        qs.append(final_data)
                        continue
            except Exception:
                pass
            
            # 모든 시도가 실패한 경우 (매우 드묾), 스킵
            continue
    
    # 질문이 부족하면 반복 사용
    while len(qs) < num_questions and len(qs) > 0:
        qs.append(qs[len(qs) % len(qs)])
    
    return qs[:num_questions] if qs else []


def grade(quiz: List[Dict[str, Any]], user_choices: List[int]) -> Dict[str, Any]:
    correct = 0
    details = []
    for i, q in enumerate(quiz):
        ai = q.get("answer_index", 0)
        uc = user_choices[i] if i < len(user_choices) else -1
        ok = int(ai == uc)
        correct += ok
        details.append({
            "question": q.get("question", ""),
            "user": uc,
            "answer": ai,
            "correct": bool(ok),
            "citation": q.get("citation", {}),
        })
    return {"score": correct, "total": len(quiz), "details": details}
