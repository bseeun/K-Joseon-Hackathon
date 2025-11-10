from __future__ import annotations

import io
import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from rag.parser import pdf_parser
from rag.embed import embed_texts
from rag.index import build_faiss_ip_index, save_index
from rag.store import (
    list_manuals,
    register_manual,
    update_meta_counts,
    save_chunks,
    save_embeddings,
    manual_paths,
    delete_manual,
)
from rag.chat import answer as rag_answer


load_dotenv()
st.set_page_config(page_title="매뉴얼 챗봇", layout="wide")


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


if not _has_api_key():
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣어주세요.")


# --- Session State ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # chat_id -> list[ {role, content} ]
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}  # chat_id -> title
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False
if "language" not in st.session_state:
    st.session_state.language = "한국어"
if "role" not in st.session_state:
    st.session_state.role = "3등 기관사"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "delete_pending" not in st.session_state:
    st.session_state.delete_pending = None  # 삭제 대기 중인 chat_id
# 페이지 간 이동 시 다이얼로그 상태 초기화
if "current_page" not in st.session_state:
    st.session_state.current_page = "app"
if st.session_state.get("current_page") != "app":
    st.session_state.show_settings = False
    st.session_state.show_upload = False
    st.session_state.current_page = "app"


def _new_chat() -> str:
    chat_id = f"chat-{len(st.session_state.conversations) + 1}"
    st.session_state.conversations[chat_id] = []
    st.session_state.chat_titles[chat_id] = "새 대화"
    st.session_state.active_chat = chat_id
    return chat_id


def _get_chat_title(chat_id: str) -> str:
    """대화 제목 반환 (첫 질문 기반)"""
    if chat_id in st.session_state.chat_titles:
        title = st.session_state.chat_titles[chat_id]
        if title != "새 대화":
            return title
    
    # 첫 번째 사용자 메시지에서 제목 생성
    conv = st.session_state.conversations.get(chat_id, [])
    for msg in conv:
        if msg.get("role") == "user":
            first_q = msg.get("content", "")
            if first_q:
                # 30자 이상이면 중간에 자르기
                if len(first_q) > 30:
                    return first_q[:15] + "..." + first_q[-12:]
                return first_q
    
    return st.session_state.chat_titles.get(chat_id, "새 대화")


def _sidebar():
    st.sidebar.title("매뉴얼 챗봇")

    # 새 채팅 버튼 크기 줄이기 (100% 대신 고정 폭 사용)
    st.sidebar.markdown("""
        <style>
        div[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            max-width: 140px !important; /* 원하는 크기로 변경 */
            width: 140px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 새 채팅 버튼
    if st.sidebar.button("새 채팅", key="new_chat_btn", type="primary"):
        _new_chat()

    st.sidebar.markdown("---")
    st.sidebar.subheader("대화 히스토리")

    # CSS 커스터마이징
    st.sidebar.markdown("""
    <style>
    /* 메뉴 버튼 고정 크기 */
    button[key*="menu_btn"] {
        min-width: 30px !important;
        max-width: 30px !important;
        width: 30px !important;
        padding: 2px 0 !important;
    }

    /* 대화 제목 버튼 말줄임 처리 */
    button[key*="chat_btn"] {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

    chats = list(st.session_state.conversations.keys())

    for cid in chats:
        title = _get_chat_title(cid)

        # 비율을 좁혀서 오른쪽 점(...) 공간 확보
        col1, col2 = st.sidebar.columns([8.5, 1.5], gap="small")

        with col1:
            if st.button(title, use_container_width=True, key=f"chat_btn_{cid}"):
                st.session_state.active_chat = cid
                st.session_state.delete_pending = None
                st.session_state.show_upload = False
                st.session_state.show_settings = False

        with col2:
            if st.button("···", key=f"menu_btn_{cid}", help="옵션"):
                st.session_state.delete_pending = None if st.session_state.delete_pending == cid else cid
                st.rerun()

        if st.session_state.delete_pending == cid:
            if st.sidebar.button("삭제", key=f"confirm_delete_{cid}", type="primary", use_container_width=True):
                del st.session_state.conversations[cid]
                if cid in st.session_state.chat_titles:
                    del st.session_state.chat_titles[cid]

                if st.session_state.active_chat == cid:
                    remaining = [c for c in chats if c != cid]
                    st.session_state.active_chat = remaining[0] if remaining else None
                st.session_state.delete_pending = None
                st.rerun()

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
        key="settings_language"
    )
    
    # 표시값을 내부값으로 변환
    language = language_internal.get(language_display_selected, "한국어")
    
    roles = ["3등 기관사", "2등 기관사", "1등 기관사", "기관장"]
    
    role = st.selectbox(
        "직급 선택",
        options=roles,
        index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
        key="settings_role"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("적용", type="primary", use_container_width=True, key="settings_apply"):
            st.session_state.language = language
            st.session_state.role = role
            st.session_state.show_settings = False
            st.rerun()
    
    with col2:
        if st.button("닫기", use_container_width=True, key="settings_close"):
            st.session_state.show_settings = False
            st.rerun()


def _topbar_upload_button():
    col1, col_upload, col_setting = st.columns([1, 0.2, 0.15])
    with col_upload:
        # chat_input이 처리 중이면 버튼 비활성화
        if st.button("소스 업로드", disabled=not _has_api_key() or st.session_state.get("chat_input_processed", False), key="topbar_upload_btn"):
            st.session_state.show_upload = True
            st.session_state.show_settings = False  # 설정 다이얼로그 닫기
            st.session_state.chat_input_processed = False  # 플래그 리셋
            st.rerun()
        if not _has_api_key():
            st.caption("API 키가 없으면 업로드/인덱싱을 사용할 수 없습니다.")
    with col_setting:
        # chat_input이 처리 중이면 버튼 비활성화
        if st.button("설정", disabled=not _has_api_key() or st.session_state.get("chat_input_processed", False), use_container_width=True, key="topbar_settings_btn"):
            st.session_state.show_settings = True
            st.session_state.show_upload = False  # 업로드 다이얼로그 닫기
            st.session_state.chat_input_processed = False  # 플래그 리셋
            st.rerun()
    
    # Settings dialog (업로드 다이얼로그가 열려있지 않을 때만, chat_input 처리 중이 아닐 때만)
    if (st.session_state.show_settings and 
        not st.session_state.show_upload and 
        not st.session_state.get("chat_input_processed", False)):
        try:
            @st.dialog("설정", width="medium")
            def _dlg():
                _settings_dialog()
            _dlg()
        except Exception:
            with st.expander("설정", expanded=True):
                _settings_dialog()

    # Modal/dialog (설정 다이얼로그가 열려있지 않을 때만, chat_input 처리 중이 아닐 때만)
    if (st.session_state.show_upload and 
        not st.session_state.show_settings and 
        not st.session_state.get("chat_input_processed", False)):
        try:

            @st.dialog("소스 업로드", width="large")
            def _dlg():
                _upload_dialog_body()
                if st.button("닫기", key="upload_dialog_close"):
                    st.session_state.show_upload = False
                    st.rerun()

            _dlg()
        except Exception:
            with st.expander("소스 업로드", expanded=True):
                _upload_dialog_body()
                if st.button("닫기", key="upload_expander_close"):
                    st.session_state.show_upload = False
                    st.rerun()


def _chat_body():
    _topbar_upload_button()

    if st.session_state.active_chat is None:
        _new_chat()

    st.markdown("## MARINOVA")

    # Render history
    for msg in st.session_state.conversations[st.session_state.active_chat]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                
                # 이미지 표시 (히스토리)
                images = msg.get("images", [])
                if images:
                    st.markdown("#### 관련 이미지")
                    for img_data in images:
                        st.caption(f"{img_data['title']} (페이지 {img_data['page']})")
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data["image_bytes"]))
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.caption("이미지 로드 실패")
                
                cites = msg.get("citations") or []
                if cites:
                    cite_texts = []
                    for c in cites:
                        cite_text = f"{c['title']} (p.{c['page']})"
                        if c.get("has_image", False):
                            cite_text += " 📷"
                        cite_texts.append(cite_text)
                    st.caption("출처: " + ", ".join(cite_texts))

    if not _has_api_key():
        st.info("OPENAI_API_KEY 설정 후 채팅을 이용할 수 있습니다.")
        return

    # Guard: no manuals
    if not list_manuals():
        st.info(
            "업로드된 매뉴얼이 없습니다. 우상단 '소스 업로드'에서 PDF를 등록해 주세요."
        )
        return

    # Input 전에 다이얼로그 상태 확인 및 닫기
    # chat_input이 실행되면 rerun이 발생하므로, 이전에 다이얼로그가 열려있으면 닫기
    if "chat_input_processed" not in st.session_state:
        st.session_state.chat_input_processed = False
    
    # Input
    prompt = st.chat_input("메뉴얼에 대해 물어보세요…")
    
    # prompt가 있으면 다이얼로그 강제로 닫기
    if prompt:
        st.session_state.show_upload = False
        st.session_state.show_settings = False
        st.session_state.chat_input_processed = True

        # 첫 번째 질문이면 제목 설정
        conv = st.session_state.conversations[st.session_state.active_chat]
        if len(conv) == 0:
            # 제목 생성 (30자 이상이면 중간에 자르기)
            if len(prompt) > 30:
                title = prompt[:15] + "..." + prompt[-12:]
            else:
                title = prompt
            st.session_state.chat_titles[st.session_state.active_chat] = title
        
        # 사용자 메시지를 먼저 표시
        st.chat_message("user").markdown(prompt)
        
        # 세션 상태에 사용자 메시지 추가
        st.session_state.conversations[st.session_state.active_chat].append(
            {"role": "user", "content": prompt}
        )
        
        with st.chat_message("assistant"):
            with st.spinner("검색 중…"):
                # 현재 대화의 이전 메시지들을 히스토리로 전달 (assistant 메시지만 제외)
                conv_history = []
                for msg in st.session_state.conversations[st.session_state.active_chat]:
                    if msg["role"] == "user":
                        conv_history.append({"role": "user", "content": msg["content"]})
                    # assistant 메시지는 너무 길 수 있으므로 제외 (RAG 결과 포함)
                
                res = rag_answer(
                    prompt, 
                    top_k=5,
                    language=st.session_state.language,
                    role=st.session_state.role,
                    conversation_history=conv_history
                )
                answer_text = res.get("answer", "")
                citations = res.get("citations", [])
                st.markdown(answer_text)
                
                # 이미지 표시
                images = res.get("images", [])
                if images:
                    st.markdown("#### 관련 이미지")
                    for img_data in images:
                        st.caption(f"{img_data['title']} (페이지 {img_data['page']})")
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(img_data["image_bytes"]))
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.caption("이미지 로드 실패")
                
                if citations:
                    cite_texts = []
                    for c in citations:
                        cite_text = f"{c['title']} (p.{c['page']})"
                        if c.get("has_image", False):
                            cite_text += " 📷"
                        cite_texts.append(cite_text)
                    st.caption("출처: " + ", ".join(cite_texts))
        st.session_state.conversations[st.session_state.active_chat].append(
            {"role": "assistant", "content": answer_text, "citations": citations, "images": res.get("images", [])}
        )
        
        # 답변 완료 후 플래그 리셋 (다음 rerun에서 다이얼로그가 정상적으로 작동하도록)
        st.session_state.chat_input_processed = False


# --- Layout ---
_sidebar()
_chat_body()
