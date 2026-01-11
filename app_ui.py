"""
Gradio UI for RAG System (Alternative to Streamlit)

This is a simpler UI using Gradio instead of needing a separate server.
"""

import sys
sys.path.insert(0, 'src')

import gradio as gr
from ccba_rag.core.rag_system import RAGSystem
from ccba_rag.core.settings import settings

# Initialize System
system = RAGSystem(verbose=True)

def query_rag(message, history):
    try:
        # Execute Query
        result = system.query(message, verbose=False)
        
        answer = result['answer']
        
        # Format Citations
        citations = ""
        contexts = result.get('contexts', [])
        if contexts:
            citations = "\n\n**Nguồn tham khảo:**\n"
            for i, c in enumerate(contexts[:5], 1):
                citations += f"{i}. {c.get('document_name', 'Unknown')}"
                if c.get('article'): citations += f", Điều {c['article']}"
                if c.get('clause'): citations += f", Khoản {c['clause']}"
                citations += "\n"
        
        # Add Stats
        stats = result.get('stats', {})
        footer = f"\n\n---\n*⏱️ Retrieval: {stats.get('retrieval_ms', 0):.0f}ms | Total: {stats.get('total_ms', 0):.0f}ms | Model: {result.get('model', 'N/A')}*"
        
        return answer + citations + footer
        
    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}"

# Define UI
with gr.Blocks(title="RAG Quy Chuẩn Xây Dựng", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏗️ Trợ Lý Quy Chuẩn & Tiêu Chuẩn Xây Dựng Việt Nam")
    gr.Markdown("Hỏi đáp về QCVN 01:2021, QCVN 06:2022, TCVN 4451:2012 và các văn bản pháp luật khác.")
    
    chatbot = gr.ChatInterface(
        query_rag,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Nhập câu hỏi của bạn (ví dụ: Chiều cao tối thiểu của tầng 1?)...", container=False, scale=7),
        submit_btn="Gửi",
        retry_btn="Thử lại",
        undo_btn="Hoàn tác",
        clear_btn="Xóa hội thoại",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
