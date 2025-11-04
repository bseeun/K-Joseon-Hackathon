from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal
import random

from openai import OpenAI

from .store import load_chunks, manual_paths
from .embed import embed_query
from .index import load_index, search as faiss_search
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


def _select_chunks(
    manual_id: str,
    chunks: List[Dict[str, Any]],
    mode: Literal["random", "topic"] = "random",
    topic: Optional[str] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    if mode == "topic" and topic:
        try:
            idx = load_index(manual_paths(manual_id)["index"])
            qv = embed_query(topic)
            _, ids = faiss_search(idx, qv, top_k=k)
            selected = [chunks[i] for i in ids if 0 <= i < len(chunks)]
            if selected:
                return selected
        except Exception:
            pass
    # fallback random
    rng = random.Random(42)
    return rng.sample(chunks, k=min(k, len(chunks)))


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
    role: Optional[str] = None,
    quiz_type: Literal["mcq", "ordering"] = "mcq",
    topic: Optional[str] = None,
    selection: Literal["random", "topic"] = "random",
) -> List[Dict[str, Any]]:
    chunks = load_chunks(manual_id)
    base_pool = _select_chunks(
        manual_id,
        chunks,
        mode=("topic" if topic and selection == "topic" else "random"),
        topic=topic,
        k=4,
    )
    ctx = _sample_context(base_pool)
    lang_instruction = _get_language_instruction(language)
    role_info = get_role_info_for_prompt(role) if role else ""

    if quiz_type == "ordering":
        prompt_parts = [
            f"아래 선박 매뉴얼 내용을 바탕으로 절차/프로세스의 올바른 순서를 묻는 순서 맞추기 퀴즈를 {lang_instruction} 만들어주세요."
        ]
    else:
        prompt_parts = [
            f"아래 선박 매뉴얼 내용을 바탕으로 객관식 퀴즈를 {lang_instruction} 만들어주세요."
        ]
    
    if role_info:
        prompt_parts.append(role_info)
        prompt_parts.append(f"퀴즈 문제는 위 직급 정보를 고려하여 해당 직급에 적합한 수준의 난이도와 내용으로 출제하세요.")
    
    if quiz_type == "ordering":
        prompt_parts.extend([
            "출력 형식은 JSON 배열이며 각 원소는 다음 키를 가집니다: \n",
            "{type: 'ordering', question: str, items_shuffled: [str, ...], correct_order: [str, ...], explanation: str, citation: {title: str, page: int}}\n",
            f"[자료]\n{ctx}\n",
            f"문항 수: {num_questions}\n",
            "요구 사항: 자료에서 실제 절차를 추출해 'correct_order'에 올바른 순서로 넣고, 'items_shuffled'는 동일 항목을 무작위로 섞어 제시하세요. 항목 수는 4~7개. explanation에 핵심 근거를 간단히 작성."
        ])
    else:
        prompt_parts.extend([
            "출력 형식은 JSON 배열이며 각 원소는 다음 키를 가집니다: \n",
            "{type: 'mcq', question: str, options: [str,str,str,str], answer_index: int, explanation: str, citation: {title: str, page: int}}\n",
            f"[자료]\n{ctx}\n",
            f"문항 수: {num_questions}\n",
            "각 문항은 정답이 되는 이유를 explanation에 2~3문장으로 간단히 설명하세요."
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
        max_tokens=2000,
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
    random.shuffle(base_pool)  # 질문 순서를 다양하게
    
    for i in range(min(num_questions, len(base_pool))):
        ch = base_pool[i]
        header = ch.get('header', '')
        content = str(ch.get('content', '')).strip()
        page = ch.get("start_page", 0)
        
        if not content:
            continue
        
        # 각 청크에 대해 개별적으로 LLM으로 퀴즈 생성
        if quiz_type == "ordering":
            chunk_prompt_parts = [
                f"아래 선박 매뉴얼 내용을 바탕으로 순서 맞추기 퀴즈를 {lang_instruction} 만들어주세요."
            ]
        else:
            chunk_prompt_parts = [
                f"아래 선박 매뉴얼 내용을 바탕으로 객관식 퀴즈를 {lang_instruction} 만들어주세요."
            ]
        
        if role_info:
            chunk_prompt_parts.append(role_info)
            chunk_prompt_parts.append(f"퀴즈 문제는 위 직급 정보를 고려하여 해당 직급에 적합한 수준의 난이도와 내용으로 출제하세요.")
        
        if quiz_type == "ordering":
            chunk_prompt_parts.extend([
                "출력 형식은 JSON 객체이며 다음 키를 가집니다: \n",
                "{type: 'ordering', question: str, items_shuffled: [str, ...], correct_order: [str, ...], explanation: str}\n",
                f"[자료]\n제목: {header}\n내용: {content[:800]}\n",
                "실제 절차 항목을 4~7개로 작성하세요. items_shuffled는 correct_order의 동일 항목을 무작위 순서로 제시하세요."
            ])
        else:
            chunk_prompt_parts.extend([
                "출력 형식은 JSON 객체이며 다음 키를 가집니다: \n",
                "{type: 'mcq', question: str, options: [str,str,str,str], answer_index: int, explanation: str}\n",
                f"[자료]\n제목: {header}\n내용: {content[:500]}\n",
                "하나의 객관식 문제만 생성하세요. answer_index는 0부터 3 사이의 정수입니다. explanation에 정답이 되는 이유를 1~2문장으로 설명."
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
                    max_tokens=800,
                )
                chunk_text = chunk_resp.choices[0].message.content or "{}"
                
                # JSON 파싱 시도
                try:
                    # JSON 코드 블록 제거 시도
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', chunk_text, re.DOTALL)
                    if json_match:
                        chunk_text = json_match.group(0)
                    
                    chunk_data = json.loads(chunk_text)
                    if quiz_type == "ordering":
                        if isinstance(chunk_data, dict) and "items_shuffled" in chunk_data and "correct_order" in chunk_data:
                            chunk_data["citation"] = {"title": header, "page": page}
                            qs.append(chunk_data)
                            success = True
                            break
                    else:
                        if isinstance(chunk_data, dict) and "question" in chunk_data and "options" in chunk_data:
                            if isinstance(chunk_data["options"], list) and len(chunk_data["options"]) == 4:
                                ans_idx = chunk_data.get("answer_index", 0)
                                if isinstance(ans_idx, int) and 0 <= ans_idx < 4:
                                    chunk_data.setdefault("explanation", "")
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
                if quiz_type == "ordering":
                    simple_prompt = (
                        f"아래 내용으로 순서 맞추기 문제 하나를 {lang_instruction} 만들어주세요.\n"
                        f"제목: {header}\n"
                        f"내용: {content[:400]}\n\n"
                        "JSON 형식으로 응답: {\"type\": \"ordering\", \"question\": \"...\", \"items_shuffled\": [\"A\", \"B\", \"C\"], \"correct_order\": [\"A\", \"B\", \"C\"], \"explanation\": \"...\"}"
                    )
                else:
                    simple_prompt = (
                        f"아래 내용으로 객관식 문제 하나를 {lang_instruction} 만들어주세요.\n"
                        f"제목: {header}\n"
                        f"내용: {content[:300]}\n\n"
                        "JSON 형식으로 응답: {\"type\": \"mcq\", \"question\": \"질문\", \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"], \"answer_index\": 0, \"explanation\": \"이유\"}"
                    )
                
                final_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "JSON 형식으로만 응답하세요."},
                        {"role": "user", "content": simple_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                )
                final_text = final_resp.choices[0].message.content or "{}"
                
                # JSON 추출 및 파싱
                import re
                json_match = re.search(r'\{[\s\S]*\}', final_text, re.DOTALL)
                if json_match:
                    final_text = json_match.group(0)
                
                final_data = json.loads(final_text)
                if quiz_type == "ordering":
                    if isinstance(final_data, dict) and "items_shuffled" in final_data and "correct_order" in final_data:
                        final_data["citation"] = {"title": header, "page": page}
                        qs.append(final_data)
                        continue
                else:
                    if isinstance(final_data, dict) and "question" in final_data and "options" in final_data:
                        if isinstance(final_data["options"], list) and len(final_data["options"]) == 4:
                            final_data.setdefault("explanation", "")
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
        qtype = q.get("type", "mcq")
        if qtype == "ordering":
            # ordering 채점은 프론트에서 순서 비교가 필요하므로 여기선 패스/표시만
            details.append({
                "question": q.get("question", ""),
                "type": "ordering",
                "citation": q.get("citation", {}),
                "correct": None,
            })
            continue
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
            "explanation": q.get("explanation", ""),
        })
    return {"score": correct, "total": len([d for d in details if d.get("type") != "ordering"]), "details": details}
