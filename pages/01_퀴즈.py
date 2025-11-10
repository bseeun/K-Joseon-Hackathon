from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import os
import tempfile

from rag.store import list_manuals, register_manual, update_meta_counts, save_chunks, save_embeddings, manual_paths, delete_manual
from rag.parser import pdf_parser
from rag.embed import embed_texts
from rag.index import build_faiss_ip_index, save_index
from rag.quiz import generate_quiz, grade

load_dotenv()
st.set_page_config(page_title="퀴즈", layout="wide")

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


if not HAS_KEY:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣어주세요.")

if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "quiz_idx" not in st.session_state:
    st.session_state.quiz_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "quiz_manual" not in st.session_state:
    st.session_state.quiz_manual = None
if "language" not in st.session_state:
    st.session_state.language = "한국어"
if "role" not in st.session_state:
    st.session_state.role = "3등 기관사"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "quiz_type" not in st.session_state:
    st.session_state.quiz_type = "mcq"  # "mcq" | "ordering"
if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = "random"  # "random" | "topic"
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "num_questions" not in st.session_state:
    st.session_state.num_questions = 5
if "ordering_answers" not in st.session_state:
    st.session_state.ordering_answers = {}  # idx -> List[str]
if "ordering_user" not in st.session_state:
    st.session_state.ordering_user = {}  # idx -> List[str] from draggable table
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

# 페이지 간 이동 시 다이얼로그 상태 초기화
if "current_page" not in st.session_state:
    st.session_state.current_page = "quiz"
if st.session_state.get("current_page") != "quiz":
    st.session_state.show_settings = False
    st.session_state.show_upload = False
    st.session_state.current_page = "quiz"


# Sidebar: manual select and upload shortcut
st.sidebar.title("퀴즈 설정")
manuals = list_manuals()
manual_opts = {m["title"]: m["id"] for m in manuals} if manuals else None
if manual_opts:
    sel_title = st.sidebar.selectbox("매뉴얼 선택", [*manual_opts.keys()])
else:
    sel_title = "(없음)"

st.sidebar.markdown("---")
st.sidebar.subheader("문항 유형")
quiz_type_label = st.sidebar.selectbox(
    "퀴즈 유형",
    options=["객관식", "순서 맞추기"],
    index=0 if st.session_state.quiz_type == "mcq" else 1,
)
st.session_state.quiz_type = "mcq" if quiz_type_label == "객관식" else "ordering"

st.sidebar.subheader("문제 생성 설정")
st.session_state.num_questions = st.sidebar.slider("문항 수", 1, 10, st.session_state.num_questions)
sel_mode_label = st.sidebar.radio("문제 생성 방식", ["무작위", "주제 기반"], index=0 if st.session_state.selection_mode == "random" else 1, horizontal=True)
st.session_state.selection_mode = "random" if sel_mode_label == "무작위" else "topic"
if st.session_state.selection_mode == "topic":
    st.session_state.topic = st.sidebar.text_input("주제 입력 (예: 조수기 정지 절차)", value=st.session_state.topic)


def _upload_dialog_body():
    st.subheader("소스 업로드")

    if not _has_api_key():
        st.info("OPENAI_API_KEY 설정 후 이용해 주세요.")
        return

    # Existing manuals
    manuals = list_manuals()
    if manuals:
        st.markdown("#### 업로드된 매뉴얼")
        for m in manuals:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"- {m['title']} (id: {m['id']})")
            with col2:
                if st.button("삭제", key=f"delete_manual_{m['id']}", help="매뉴얼 삭제"):
                    if delete_manual(m['id']):
                        st.success(f"'{m['title']}' 매뉴얼이 삭제되었습니다.")
                        st.rerun()
                    else:
                        st.error("매뉴얼 삭제에 실패했습니다.")
    else:
        st.caption("아직 업로드된 매뉴얼이 없습니다.")

    st.markdown("---")
    file = st.file_uploader(
        "PDF 매뉴얼 업로드", type=["pdf"], accept_multiple_files=False
    )

    if file is not None:
        title = st.text_input("매뉴얼 제목", value=os.path.splitext(file.name)[0])
        proceed = st.button("업로드 및 인덱싱 시작", type="primary")
        if proceed:
            with st.status("인덱싱 중...", expanded=True) as status:
                try:
                    # Save uploaded PDF to temp
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    st.write("1/4 PDF 저장 완료")

                    # Register manual -> copy to data folder
                    meta = register_manual(title, tmp_path)
                    mid = meta["id"]
                    st.write(f"2/4 매뉴얼 등록 완료(id: {mid})")

                    # Parse -> chunks
                    chunks = pdf_parser(manual_paths(mid)["pdf"])
                    save_chunks(mid, chunks)
                    st.write(f"3/4 파싱 완료, 청크 수: {len(chunks)}")

                    # Embed -> index
                    docs = [
                        f"제목: {c.get('header','')}, 내용: {c.get('content','')}"
                        for c in chunks
                    ]
                    emb = embed_texts(docs)
                    save_embeddings(mid, emb)
                    idx = build_faiss_ip_index(emb)
                    save_index(idx, manual_paths(mid)["index"])

                    # Update meta
                    try:
                        import fitz

                        with fitz.open(manual_paths(mid)["pdf"]) as d:
                            update_meta_counts(mid, d.page_count, len(chunks))
                    except Exception:
                        update_meta_counts(mid, None, len(chunks))

                    status.update(label="완료", state="complete")
                    st.success("업로드/인덱싱이 완료되었습니다.")
                    st.rerun()  # 매뉴얼 목록 새로고침
                except Exception as e:
                    status.update(label="실패", state="error")
                    st.error(f"오류: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


def _settings_dialog():
    """언어 및 직급 설정 다이얼로그"""
    st.subheader("설정")
    
    # 언어 표시용과 내부 값 매핑
    language_display = {
        "한국어": "한국어",
        "영어": "English",
        "중국어": "中文",
        "일본어": "日本語"
    }
    language_internal = {v: k for k, v in language_display.items()}  # 역매핑
    
    languages_display = ["한국어", "English", "中文", "日本語"]
    languages_internal = ["한국어", "영어", "중국어", "일본어"]
    
    # 현재 선택된 언어의 표시값 찾기
    current_lang_display = language_display.get(st.session_state.language, "한국어")
    current_index = languages_display.index(current_lang_display) if current_lang_display in languages_display else 0
    
    language_display_selected = st.selectbox(
        "언어 선택",
        options=languages_display,
        index=current_index,
        key="quiz_settings_language"
    )
    
    # 표시값을 내부값으로 변환
    language = language_internal.get(language_display_selected, "한국어")
    
    roles = ["3등 기관사", "2등 기관사", "1등 기관사", "기관장"]
    
    role = st.selectbox(
        "직급 선택",
        options=roles,
        index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
        key="quiz_settings_role"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("적용", type="primary", use_container_width=True, key="quiz_settings_apply"):
            st.session_state.language = language
            st.session_state.role = role
            st.session_state.show_settings = False
            st.rerun()
    
    with col2:
        if st.button("닫기", use_container_width=True, key="quiz_settings_close"):
            st.session_state.show_settings = False
            st.rerun()


# 상단 설정 버튼
col_title, col_upload, col_setting = st.columns([1, 0.2, 0.15])
with col_upload:
    if st.button("소스 업로드", disabled=not HAS_KEY, key="upload_button"):
        st.session_state.show_upload = True
        st.session_state.show_settings = False  # 설정 다이얼로그 닫기
    if not HAS_KEY:
        st.caption("API 키가 없으면 업로드/인덱싱을 사용할 수 없습니다.")
with col_setting:
    if st.button("설정", disabled=not HAS_KEY, use_container_width=True, key="settings_button"):
        st.session_state.show_settings = True
        st.session_state.show_upload = False  # 업로드 다이얼로그 닫기

# Settings dialog (업로드 다이얼로그가 열려있지 않을 때만)
if st.session_state.show_settings and not st.session_state.show_upload:
    try:
        @st.dialog("설정", width="medium")
        def _settings_dlg():
            _settings_dialog()
        _settings_dlg()
    except Exception:
        with st.expander("설정", expanded=True):
            _settings_dialog()

# Upload dialog (설정 다이얼로그가 열려있지 않을 때만)
if st.session_state.show_upload and not st.session_state.show_settings:
    try:
        @st.dialog("소스 업로드", width="large")
        def _dlg():
            _upload_dialog_body()
            if st.button("닫기", key="quiz_upload_dialog_close"):
                st.session_state.show_upload = False
                st.rerun()
        _dlg()
    except Exception:
        with st.expander("소스 업로드", expanded=True):
            _upload_dialog_body()
            if st.button("닫기", key="quiz_upload_expander_close"):
                st.session_state.show_upload = False
                st.rerun()

col_left, col_main = st.columns([1, 3])
with col_main:
    title_map = {"mcq": "객관식 퀴즈", "ordering": "순서 맞추기 퀴즈"}
    st.markdown(f"## {title_map.get(st.session_state.quiz_type, '퀴즈')}")

    if not manuals:
        st.info("먼저 메인 페이지에서 PDF 매뉴얼을 업로드하세요.")
    else:
        manual_id = manual_opts.get(sel_title) if manual_opts else None
        if manual_id and st.session_state.quiz_manual != manual_id:
            st.session_state.quiz_manual = manual_id
            st.session_state.quiz = []
            st.session_state.quiz_idx = 0
            st.session_state.answers = []

        if not st.session_state.quiz:
            if st.button("퀴즈 생성", type="primary", disabled=not HAS_KEY, key="generate_quiz"):
                with st.spinner("문항 생성 중…"):
                    st.session_state.quiz = generate_quiz(
                        manual_id,
                        num_questions=st.session_state.num_questions,
                        language=st.session_state.language,
                        role=st.session_state.role,
                        quiz_type=st.session_state.quiz_type,
                        topic=(st.session_state.topic if st.session_state.selection_mode == "topic" else None),
                        selection=st.session_state.selection_mode,
                    )
                    st.session_state.quiz_idx = 0
                    if st.session_state.quiz_type == "mcq":
                        st.session_state.answers = [-1] * len(st.session_state.quiz)
                    else:
                        st.session_state.answers = []
                        st.session_state.ordering_answers = {}

        if st.session_state.quiz:
            idx = st.session_state.quiz_idx
            q = st.session_state.quiz[idx]
            st.write(f"문제 {idx+1}/{len(st.session_state.quiz)}")
            st.markdown(f"**{q['question']}**")

            if q.get("type", "mcq") == "mcq":
                choice = st.radio(
                    "정답을 선택하세요",
                    options=list(range(len(q["options"]))),
                    format_func=lambda i: q["options"][i],
                    index=(
                        st.session_state.answers[idx]
                        if st.session_state.answers[idx] >= 0
                        else 0
                    ),
                    key=f"q_{idx}_choice",
                )
                st.session_state.answers[idx] = choice
            else:
                # ordering UI: 커뮤니티 컴포넌트 사용하여 드래그 정렬
                items = q.get("items_shuffled", [])
                try:
                    from streamlit_sortables import sort_items  # type: ignore
                    ordered = sort_items(items, direction="vertical", key=f"ord_dnd_{idx}")
                    st.session_state.ordering_user[idx] = ordered or items
                except Exception:
                    # Fallback: 편집 불가 테이블(순서 변경 불가) + 안내
                    df = pd.DataFrame({"항목": items})
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    st.info("드래그 UI를 사용하려면 'streamlit-sortables' 설치가 필요합니다: pip install streamlit-sortables")
                    st.session_state.ordering_user[idx] = items

            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button("이전", disabled=idx == 0, key=f"prev_{idx}"):
                    st.session_state.quiz_idx = max(0, idx - 1)
                    st.rerun()
            with cols[1]:
                if st.button(
                    "다음",
                    type="primary",
                    disabled=idx == len(st.session_state.quiz) - 1,
                    key=f"next_{idx}"
                ):
                    st.session_state.quiz_idx = min(
                        len(st.session_state.quiz) - 1, idx + 1
                    )
                    st.rerun()

            st.markdown("---")
            if st.button("결과 보기", type="secondary", disabled=not HAS_KEY, key="show_result"):
                if st.session_state.quiz_type == "mcq":
                    res = grade(st.session_state.quiz, st.session_state.answers)
                else:
                    # ordering 채점: 사용자 순서와 정답 순서 비교
                    details = []
                    correct_count = 0
                    for qi, qq in enumerate(st.session_state.quiz):
                        if qq.get("type", "ordering") != "ordering":
                            continue
                        user_seq = st.session_state.ordering_user.get(qi, [])
                        if not user_seq or len(user_seq) != len(qq.get("correct_order", [])):
                            ok = False
                        else:
                            ok = user_seq == qq.get("correct_order", [])
                        correct_count += int(ok)
                        details.append({
                            "question": qq.get("question", ""),
                            "type": "ordering",
                            "user_order": user_seq,
                            "correct_order": qq.get("correct_order", []),
                            "correct": ok,
                            "citation": qq.get("citation", {}),
                            "explanation": qq.get("explanation", ""),
                        })
                    res = {"score": correct_count, "total": len(details), "details": details}
                st.session_state.quiz_result = res
                st.rerun()

        if "quiz_result" in st.session_state:
            res = st.session_state.quiz_result
            st.success(f"점수: {res['score']} / {res['total']}")
            for i, d in enumerate(res["details"], 1):
                if d.get("type") == "ordering":
                    st.write(f"{i}. {'✅' if d['correct'] else '❌'} | 출처: {d['citation'].get('title','')} (p.{d['citation'].get('page','?')})")
                    st.markdown("- 내가 고른 순서:")
                    st.markdown("  - " + " → ".join([it for it in d.get("user_order", []) if it]))
                    st.markdown("- 정답 순서:")
                    st.markdown("  - " + " → ".join(d.get("correct_order", [])))
                    if d.get("explanation"):
                        st.markdown(d["explanation"])
                else:
                    st.write(
                        f"{i}. {'✅' if d['correct'] else '❌'} 정답: {d['answer']+1}, 선택: {d['user']+1} | 출처: {d['citation'].get('title','')} (p.{d['citation'].get('page','?')})"
                    )
                    if d.get("explanation"):
                        st.markdown(d["explanation"])
