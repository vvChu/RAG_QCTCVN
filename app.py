import streamlit as st
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from ccba_rag.client import APIClient

# --- CONFIGURATION ---
client = APIClient(base_url="http://127.0.0.1:8000")
API_URL = client.base_url # Keep for legacy URL construction if needed
st.set_page_config(
    page_title="Trợ Lý Quy Chuẩn Xây Dựng",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem;}
    .chat-container {border-radius: 10px; background-color: #F3F4F6; padding: 20px;}
    .citation-box {border-left: 4px solid #3B82F6; padding: 10px; background-color: #EFF6FF; margin-bottom: 10px; font-size: 0.9rem;}
    .citation-title {font-weight: bold; color: #1E40AF;}
    .stChatMessage {background-color: transparent;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Cấu hình")
    
    # Citation View Mode
    citation_mode = st.radio("Hiển thị trích dẫn:", ["Dạng Thẻ", "Dạng Bảng"], horizontal=True)

    st.divider()
    
    # Server Status
    try:
        status = client.get_status()
        if status['status'] == 'ready':
            st.success(f"🟢 Server: Online ({status.get('collection', 'unknown')})")
            if 'stats' in status and 'chunk_count' in status['stats']:
                    st.caption(f"📊 Đã index: {status['stats']['chunk_count']} chunks")
        else:
            st.warning("🟡 Server: Initializing...")
    except Exception as e:
        st.error(f"🔴 Server: Offline")
        # st.error(f"Debug: {e}") # Optional debug
        st.error("🔴 Server: Offline")
        st.info("Vui lòng chạy `python server.py`")

    st.divider()
    
    top_k = st.slider("Top K (Retrieval)", 10, 200, 30, 10, help="Số lượng tài liệu tìm kiếm thô ban đầu.")
    top_n = st.slider("Top N (Final)", 1, 20, 5, 1, help="Số lượng tài liệu tốt nhất được chọn để trả lời.")
    
    use_reranker = st.checkbox("Kích hoạt Reranker (Chậm hơn, chính xác hơn)", value=True)
    use_expansion = st.toggle("Mở rộng câu hỏi (AI)", value=False, help="Dùng AI để tìm thêm từ khóa liên quan.")
    
    # Document Filter
    st.divider()
    st.markdown("### 📂 Bộ lọc tài liệu")
    try:
        available_docs = client.get_documents()
    except:
        available_docs = []
        available_docs = []
    
    selected_docs = st.multiselect(
        "Chỉ tìm trong:",
        options=available_docs,
        placeholder="Tất cả tài liệu"
    )
    
    st.divider()
    st.markdown("### 📖 Hướng dẫn")
    st.markdown("""
    1. Nhập câu hỏi về QCVN/TCVN.
    2. Hệ thống sẽ tìm kiếm và trả lời.
    3. Kiểm tra nguồn trích dẫn ở cột phải (nếu có) hoặc dưới câu trả lời.
    """)

# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🏗️ Trợ Lý AI Quy Chuẩn Xây Dựng</div>', unsafe_allow_html=True)

# Create Tabs
tab_chat, tab_data = st.tabs(["💬 Chat Chuyên Gia", "📂 Quản lý Dữ liệu"])

with tab_data:
    st.header("Nạp dữ liệu mới")
    uploaded_file = st.file_uploader("Chọn file PDF hoặc DOCX", type=['pdf', 'docx'])
    
    if uploaded_file:
        if st.button("Upload & Index"):
            with st.spinner("Đang tải lên..."):
                try:
                    resp = client.upload_file(uploaded_file, uploaded_file.name, uploaded_file.type)
                    st.success(f"✅ {resp['message']}")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")

    st.divider()
    st.header("Đồng bộ hóa")
    st.info("Sử dụng khi bạn copy file trực tiếp vào thư mục `data/documents`.")
    if st.button("🔄 Chạy Smart Sync"):
        try:
            resp = client.trigger_indexing()
            st.success("✅ Đã kích hoạt tiến trình đồng bộ ngầm.")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                citations = message["citations"]
                if citations:
                    with st.expander(f"📚 {len(citations)} Tài liệu tham chiếu", expanded=False):
                        if citation_mode == "Dạng Bảng":
                            # Table View
                            import pandas as pd
                            data = []
                            for c in citations:
                                data.append({
                                    "Tài liệu": c['document'],
                                    "Trang": c.get('page', 1),
                                    "Điều/Khoản": c['section'],
                                    "Nội dung": c['content']
                                })
                            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                        else:
                            # Card View
                            for i, cite in enumerate(citations, 1):
                                pdf_url = f"{API_URL}/files/{cite['document']}#page={cite.get('page', 1)}"
                                
                                col_text, col_link = st.columns([4, 1])
                                with col_text:
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-title">[{i}] {cite['document']} (Trang {cite.get('page', '?')})</div>
                                        <div>📍 {cite['section']}</div>
                                        <div style="margin-top: 5px; color: #4B5563;"><em>"{cite['content']}"</em></div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col_link:
                                    st.markdown(f'''
                                    <a href="{pdf_url}" target="_blank" style="
                                        display: inline-block;
                                        padding: 10px 15px;
                                        background-color: #DC2626;
                                        color: white;
                                        text-decoration: none;
                                        border-radius: 5px;
                                        font-weight: bold;
                                        font-size: 0.8rem;
                                        text-align: center;
                                    ">
                                    📄 Xem PDF
                                    </a>
                                    ''', unsafe_allow_html=True)
            
            # Show feedback buttons for assistant messages if they don't have feedback yet
            # (Note: simple implementation, usually we track feedback ID)
            if message["role"] == "assistant" and message == st.session_state.messages[-1]:
                 col1, col2, col3 = st.columns([1, 1, 10])
                 # Buttons logic handled in response loop below, this is just for display history?
                 # Actually history buttons won't work well because of rerun. 
                 # Let's keep buttons only for the FRESH response loop.
                 pass

    # Chat Input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                start_time = time.time()
                with st.spinner("Đang tra cứu dữ liệu..."):
                    # Format history for API
                    api_history = [
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages[:-1] # Exclude current user prompt which is appended above
                    ]
                    
                    filters = None
                    if selected_docs:
                        filters = {"document_name": selected_docs}
                    
                    data = client.query(
                        query=prompt,
                        history=api_history,
                        filters=filters,
                        top_k=top_k,
                        top_n=top_n,
                        use_expansion=use_expansion,
                        use_reranker=use_reranker
                    )
                    
                if data:
                    full_response = data['answer']
                    citations = data['citations']
                    stats = data['stats']
                    
                    # Streaming effect
                    message_placeholder.markdown(full_response)
                    
                    # Stats footer
                    total_time = (time.time() - start_time) * 1000
                    st.caption(f"⏱️ {total_time:.0f}ms (Retrieval: {stats.get('retrieval_ms', 0):.0f}ms) | Model: {stats.get('model', 'unknown')}")
                    
                    # Faithfulness Check (if available)
                    if 'faithfulness_check' in data:
                        check = data['faithfulness_check']
                        if isinstance(check, dict):
                            if check.get('is_faithful'):
                                st.success(f"✅ Faithfulness Verified: {check.get('reason')}")
                            else:
                                st.warning(f"⚠️ Potential Hallucination: {check.get('reason')}")
                                
                    # Debug Mode
                    with st.expander("🛠️ Debug Info"):
                        st.json(stats)
                        if 'prompt' in data:
                            st.text_area("Prompt Used", data['prompt'], height=200)
                    
                    # Citations Expander
                    if citations:
                        with st.expander(f"📚 {len(citations)} Tài liệu tham chiếu", expanded=True):
                            for i, cite in enumerate(citations, 1):
                                pdf_url = f"{API_URL}/files/{cite['document']}#page={cite.get('page', 1)}"
                                
                                col_text, col_link = st.columns([4, 1])
                                with col_text:
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-title">[{i}] {cite['document']} (Trang {cite.get('page', '?')})</div>
                                        <div>📍 {cite['section']}</div>
                                        <div style="margin-top: 5px; color: #4B5563;"><em>"{cite['content']}"</em></div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Highlight Preview
                                    if cite.get('bbox') and isinstance(cite['bbox'], list):
                                        bbox_str = ",".join(map(str, cite['bbox']))
                                        render_url = f"{API_URL}/render/{cite['document']}/{cite.get('page', 1)}?bbox={bbox_str}"
                                        st.image(render_url, caption=f"Trang {cite.get('page', 1)} (Preview)", use_column_width=True)
                                    
                                with col_link:
                                    st.markdown(f'''
                                    <a href="{pdf_url}" target="_blank" style="
                                        display: inline-block;
                                        padding: 10px 15px;
                                        background-color: #DC2626;
                                        color: white;
                                        text-decoration: none;
                                        border-radius: 5px;
                                        font-weight: bold;
                                        font-size: 0.8rem;
                                        text-align: center;
                                    ">
                                    📄 Xem PDF
                                    </a>
                                    ''', unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "citations": citations,
                        "model": stats['model']
                    })
                    
                    # Feedback Buttons (for the latest response)
                    col1, col2, col3 = st.columns([1, 1, 10])
                    with col1:
                        if st.button("👍", key=f"like_{len(st.session_state.messages)}", help="Hữu ích"):
                            try:
                                client.send_feedback(prompt, full_response, "positive", stats['model'])
                                st.toast("Cảm ơn phản hồi của bạn!", icon="✅")
                            except:
                                st.toast("Lỗi gửi phản hồi", icon="❌")
                    with col2:
                        if st.button("👎", key=f"dislike_{len(st.session_state.messages)}", help="Chưa tốt"):
                            try:
                                client.send_feedback(prompt, full_response, "negative", stats['model'])
                                st.toast("Chúng tôi sẽ cải thiện!", icon="🙏")
                            except:
                                st.toast("Lỗi gửi phản hồi", icon="❌")
                    
                else:
                    error_msg = f"Error: {response.text}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"❌ Đã xảy ra lỗi: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
