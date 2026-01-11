"""
Prompt Management for the CCBA RAG System

Loads prompts from config/prompts.yaml and provides formatting utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ccba_rag.core.settings import settings


def _load_prompts() -> Dict[str, str]:
    """Load prompts from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "prompts.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # Fallback to built-in prompts
    return DEFAULT_PROMPTS


# Built-in fallback prompts
DEFAULT_PROMPTS = {
    "system_instruction": """[ROLE DEFINITION]
Bạn là một Chuyên gia cao cấp về Quy chuẩn & Tiêu chuẩn Xây dựng tại Việt Nam (CCBA Expert). Nhiệm vụ của bạn là tư vấn kỹ thuật chính xác dựa trên bằng chứng thực tế được cung cấp.

[METHODOLOGY - RASE]
Để đảm bảo tính chính xác và đầy đủ, hãy áp dụng tư duy RASE khi phân tích các điều khoản:
1. Requirement (Yêu cầu): Quy định bắt buộc là gì? (Phải/Không được/Cho phép)
2. Applicability (Phạm vi): Quy định này áp dụng cho đối tượng nào?
3. Selection (Điều kiện): Trong hoàn cảnh/điều kiện cụ thể nào?
4. Exception (Ngoại lệ): Có trường hợp nào được miễn trừ không?

[INSTRUCTIONS]
1. Chỉ sử dụng thông tin trong phần [CONTEXT DATA] để trả lời. Nếu không có thông tin, hãy nói "Quy chuẩn hiện tại trong cơ sở dữ liệu chưa đề cập rõ vấn đề này".
2. Trích dẫn nguồn cụ thể (Tên QCVN, Điều, Khoản) cho từng nhận định quan trọng.
3. Khi giải thích quy định, hãy làm rõ Phạm vi áp dụng và các Ngoại lệ (nếu có).
4. Phong cách trả lời: Chuyên nghiệp, Khách quan, Gãy gọn, tuân thủ văn phong kỹ thuật.
5. KHÔNG tự suy diễn hoặc thêm thông tin bên ngoài ngữ cảnh được cung cấp.
6. Nếu câu trả lời chứa dữ liệu dạng bảng (ví dụ: các thông số kỹ thuật, phân loại), hãy trình bày dưới dạng bảng Markdown để dễ đọc.
""",
    "rag_prompt_template": """{system_instruction}

[CONTEXT DATA - KHÔNG ĐƯỢC BỊA ĐẶT]
Dưới đây là các trích dẫn chính xác từ Quy chuẩn/Tiêu chuẩn (được xếp hạng theo độ liên quan):

{contexts}

[USER QUERY]
Câu hỏi: "{query}"

[ANSWER]
""",
    "context_format": """---
[Nguồn {rank}] {document_name} - {article}
Độ liên quan: {score:.2f}

{text}
---""",
}


class PromptManager:
    """
    Manages prompt templates and formatting for RAG generation.
    """

    def __init__(self):
        self._prompts = _load_prompts()

    @property
    def system_instruction(self) -> str:
        """Get the system instruction prompt."""
        return self._prompts.get("system_instruction", DEFAULT_PROMPTS["system_instruction"])

    @property
    def rag_template(self) -> str:
        """Get the RAG prompt template."""
        return self._prompts.get("rag_prompt_template", DEFAULT_PROMPTS["rag_prompt_template"])

    @property
    def context_format(self) -> str:
        """Get the context formatting template."""
        return self._prompts.get("context_format", DEFAULT_PROMPTS["context_format"])

    def format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts into a string for the prompt.

        Args:
            contexts: List of context dicts with keys:
                - text: Content text
                - document_name: Document name
                - article: Article number (optional)
                - rerank_score or retrieval_score: Relevance score

        Returns:
            Formatted contexts string
        """
        if not contexts:
            return "Không có ngữ cảnh liên quan được tìm thấy."

        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            # Get score (prefer rerank_score over retrieval_score)
            score = ctx.get("rerank_score", ctx.get("retrieval_score", 0.0))

            # Build article reference
            article_parts = []
            if ctx.get("article"):
                article_parts.append(f"Điều {ctx['article']}")
            if ctx.get("clause"):
                article_parts.append(f"Khoản {ctx['clause']}")
            article_ref = ", ".join(article_parts) if article_parts else "N/A"

            # Format using template
            formatted = self.context_format.format(
                rank=i,
                document_name=ctx.get("document_name", "Unknown"),
                article=article_ref,
                score=score,
                text=ctx.get("text", "")[:2000]  # Truncate for safety
            )
            formatted_parts.append(formatted)

        return "\n\n".join(formatted_parts)

    def build_rag_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the full RAG prompt for generation.

        Args:
            query: User question
            contexts: Retrieved contexts
            history: Optional conversation history

        Returns:
            Complete prompt string
        """
        formatted_contexts = self.format_contexts(contexts)

        prompt = self.rag_template.format(
            system_instruction=self.system_instruction,
            contexts=formatted_contexts,
            query=query
        )

        # Add conversation history if provided
        if history:
            history_text = "\n[CONVERSATION HISTORY]\n"
            for msg in history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role.upper()}: {content}\n"
            prompt = history_text + "\n" + prompt

        return prompt


# Global instance
prompt_manager = PromptManager()
