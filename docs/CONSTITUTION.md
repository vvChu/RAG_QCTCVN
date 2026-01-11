# RAG System Constitution - Nguyên tắc Cốt lõi

## Nguyên tắc 1: Tính Chính xác Thực tế (Factual Accuracy)
**Ưu tiên tuyệt đối**: Mọi câu trả lời phải có căn cứ thực tế từ tài liệu đã truy xuất.

- Bắt buộc sử dụng chiến lược **Retrieval-first (R+G)** 
- Gemini CLI chỉ được phép trả lời dựa trên ngữ cảnh đã cung cấp
- Nếu không tìm thấy thông tin liên quan, phải thông báo rõ ràng thay vì suy đoán
- Cấm hoàn toàn việc tạo ra thông tin không có trong tài liệu gốc

## Nguyên tắc 2: Hybrid Retrieval bắt buộc
**Yêu cầu kỹ thuật**: Phải sử dụng kết hợp Dense và Sparse Retrieval.

- Model: BAAI/bge-m3 với khả năng Multi-Functionality
- Trọng số cân bằng: 50% Dense + 50% Sparse
- Dense: Tìm kiếm ngữ nghĩa sâu
- Sparse: Tìm kiếm từ khóa chính xác (BM25-like)
- Mục tiêu: Tối ưu hóa cả recall và precision

## Nguyên tắc 3: Phân đoạn theo Cấu trúc Logic
**Bảo toàn ngữ cảnh**: Chunking phải theo cấu trúc tự nhiên của văn bản pháp luật.

- Phân đoạn theo đơn vị: Chương → Điều → Khoản
- Tránh cắt đứt giữa các đơn vị logic
- Tận dụng khả năng xử lý 8192 tokens của BGE-M3
- Metadata phải ghi rõ: Tên tài liệu, Chương, Điều, Khoản
- Mục tiêu: Giữ nguyên Multi-Granularity context

## Nguyên tắc 4: Trích dẫn Nguồn Chi tiết (Grounding)
**Tính minh bạch**: Mọi thông tin phải có nguồn gốc rõ ràng.

- Trích dẫn đến cấp độ: **Điều X, Khoản Y**
- Format: `[Tên Văn bản - Điều X, Khoản Y]`
- Hiển thị ngữ cảnh trích dẫn trước khi trả lời
- Cho phép người dùng xác minh thông tin
- Tăng độ tin cậy và trách nhiệm giải trình của hệ thống

---

**Ghi chú**: Bốn nguyên tắc này là không thể thương lượng và phải được tuân thủ nghiêm ngặt trong suốt quá trình phát triển và vận hành hệ thống RAG.
