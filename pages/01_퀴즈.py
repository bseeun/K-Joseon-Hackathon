from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from rag.store import list_manuals
from rag.quiz import generate_quiz, grade

load_dotenv()
st.set_page_config(page_title="퀴즈", layout="wide")

import os

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))
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


# Sidebar: manual select and upload shortcut
st.sidebar.title("퀴즈 설정")
manuals = list_manuals()
manual_opts = {m["title"]: m["id"] for m in manuals} if manuals else None
if manual_opts:
    sel_title = st.sidebar.selectbox("매뉴얼 선택", [*manual_opts.keys()])
else:
    sel_title = "(없음)"

st.sidebar.markdown("---")
if st.sidebar.button("소스 업로드", use_container_width=True):
    try:
        st.switch_page("app.py")
    except Exception:
        st.sidebar.info("메인에서 '소스 업로드'를 이용하세요.")


def _settings_dialog():
    """언어 및 직급 설정 다이얼로그"""
    st.subheader("설정")
    
    languages = ["한국어", "영어", "중국어", "일본어"]
    roles = ["3등 기관사", "2등 기관사", "1등 기관사", "기관장"]
    
    language = st.selectbox(
        "언어 선택",
        options=languages,
        index=languages.index(st.session_state.language) if st.session_state.language in languages else 0,
        key="settings_language"
    )
    
    role = st.selectbox(
        "직급 선택",
        options=roles,
        index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
        key="settings_role"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("적용", type="primary", use_container_width=True):
            st.session_state.language = language
            st.session_state.role = role
            st.session_state.show_settings = False
    
    with col2:
        if st.button("닫기", use_container_width=True):
            st.session_state.show_settings = False


# 상단 설정 버튼
col_title, col_setting = st.columns([1, 0.15])
with col_setting:
    if st.button("⚙️ 설정", disabled=not HAS_KEY, use_container_width=True):
        st.session_state.show_settings = True

# Settings dialog
if st.session_state.show_settings:
    try:
        @st.dialog("설정", width="medium")
        def _dlg():
            _settings_dialog()
        _dlg()
    except Exception:
        with st.expander("설정", expanded=True):
            _settings_dialog()

col_left, col_main = st.columns([1, 3])
with col_main:
    st.markdown("## 객관식 퀴즈")

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
            if st.button("퀴즈 생성", type="primary", disabled=not HAS_KEY):
                with st.spinner("문항 생성 중…"):
                    st.session_state.quiz = generate_quiz(
                        manual_id, 
                        num_questions=5,
                        language=st.session_state.language,
                        role=st.session_state.role
                    )
                    st.session_state.quiz_idx = 0
                    st.session_state.answers = [-1] * len(st.session_state.quiz)

        if st.session_state.quiz:
            idx = st.session_state.quiz_idx
            q = st.session_state.quiz[idx]
            st.write(f"문제 {idx+1}/{len(st.session_state.quiz)}")
            st.markdown(f"**{q['question']}**")
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

            # Save choice
            st.session_state.answers[idx] = choice

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
                res = grade(st.session_state.quiz, st.session_state.answers)
                st.session_state.quiz_result = res
                st.rerun()

        if "quiz_result" in st.session_state:
            res = st.session_state.quiz_result
            st.success(f"점수: {res['score']} / {res['total']}")
            for i, d in enumerate(res["details"], 1):
                st.write(
                    f"{i}. {'✅' if d['correct'] else '❌'} 정답: {d['answer']+1}, 선택: {d['user']+1} | 출처: {d['citation'].get('title','')} (p.{d['citation'].get('page','?')})"
                )
