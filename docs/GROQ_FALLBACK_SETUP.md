# Groq Fallback LLM Integration Guide

## 📋 Overview

Hệ thống RAG giờ đây hỗ trợ **Groq** làm LLM dự phòng (fallback) khi Gemini API gặp sự cố (503 overload, rate limit, outage).

**Cơ chế hoạt động:**

- **Primary**: Gemini 2.5 Flash (mặc định)
- **Fallback**: Llama 3.3 70B (qua Groq API)
- Tự động chuyển đổi khi Gemini trả về lỗi hoặc response rỗng

---

## 🚀 Setup

### 1. Cài đặt Dependencies

```powershell
pip install groq
```

Hoặc update toàn bộ:

```powershell
pip install -r requirements.txt
```

### 2. Lấy Groq API Key

1. Truy cập: <https://console.groq.com/keys>
2. Tạo API key mới
3. Copy key (format: `gsk_...`)

### 3. Cấu hình Environment Variables

Thêm vào `.env`:

```env
# Primary LLM (Gemini)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash

# Fallback LLM (Groq) - OPTIONAL
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

**Lưu ý**: Nếu không có `GROQ_API_KEY`, hệ thống vẫn hoạt động bình thường với chỉ Gemini.

---

## 🧪 Test Fallback

Chạy script test tích hợp:

```powershell
python test_groq_fallback.py
```

**Expected output:**

```
✅ All tests passed! Groq fallback is ready.
```

Test bao gồm:

1. Direct Groq generation (verify API connection)
2. Fallback pipeline (Gemini fail → Groq answers)

---

## 💻 Usage

### CLI Mode

```powershell
python src/main.py --mode query --question "Chiều cao tối thiểu của tầng 1?"
```

Nếu Gemini fail → tự động chuyển sang Groq.

### Python API

```python
from src.main import RAGSystem

system = RAGSystem(mode='query')
result = system.query("Khoảng cách phòng cháy giữa 2 tòa nhà?")

# Check which model answered
print(f"Primary: {result['stats']['primary_model']}")
print(f"Fallback used: {result['stats']['used_fallback']}")
print(f"Final model: {result['stats']['final_model']}")
```

---

## 📊 Fallback Logic

**Trigger conditions** (tự động activate Groq):

1. Gemini trả về exception
2. Response chứa "Error generating response"
3. Response chứa "503" hoặc "overloaded"
4. Response trống

**Response structure:**

```json
{
  "answer": "Câu trả lời...",
  "stats": {
    "primary_model": "gemini-2.5-flash",
    "fallback_model": "llama-3.3-70b-versatile",
    "used_fallback": true,
    "final_model": "llama-3.3-70b-versatile"
  }
}
```

---

## 🎯 Groq Model Options

Các model khả dụng trên Groq (fast inference):

| Model | Best For | Speed |
|-------|----------|-------|
| `llama-3.3-70b-versatile` | General Q&A, Vietnamese | ⚡⚡⚡ |
| `llama-3.1-70b-versatile` | Balanced performance | ⚡⚡⚡ |
| `mixtral-8x7b-32768` | Long context (32k tokens) | ⚡⚡ |
| `gemma-7b-it` | Fast, lightweight | ⚡⚡⚡⚡ |

**Khuyến nghị**: `llama-3.3-70b-versatile` (default) cho văn bản pháp luật tiếng Việt.

---

## ⚙️ Advanced Configuration

### Custom Fallback Chain

```python
from src.groq_generator import GroqGenerator
from src.generator import GeminiGenerator
from src.pipeline import FallbackRAGPipeline

# Setup custom generators
primary = GeminiGenerator(api_key='...', model='gemini-2.5-flash')
fallback = GroqGenerator(api_key='...', model='mixtral-8x7b-32768')

# Custom pipeline
pipeline = FallbackRAGPipeline(
    retriever=retriever,
    primary_generator=primary,
    fallback_generator=fallback
)
```

### Force Groq Mode (Testing)

```python
# Disable Gemini temporarily
os.environ.pop('GEMINI_API_KEY', None)
# Now only Groq will work
```

---

## 📈 Performance Comparison

| Model | Latency | Vietnamese Quality | Cost |
|-------|---------|-------------------|------|
| Gemini 2.5 Flash | ~2-5s | ⭐⭐⭐⭐⭐ | $ |
| Llama 3.3 70B (Groq) | ~1-3s | ⭐⭐⭐⭐ | $$ |

**Trade-offs:**

- Groq: Nhanh hơn (~30-50% faster inference)
- Gemini: Chất lượng tiếng Việt tốt hơn một chút

---

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY not found"

**Solution**: Set env variable:

```powershell
$env:GROQ_API_KEY = 'gsk_...'
```

### Fallback không activate

**Check:**

1. Gemini có đang trả lời bình thường?
2. Log có chứa "used_fallback: False"?
3. Thử force fail Gemini (dùng invalid key)

### Groq trả về 429 (rate limit)

**Solution**: Groq free tier có giới hạn:

- 30 requests/minute
- Upgrade tài khoản hoặc thêm delay giữa các query

---

## 📝 Next Steps

1. ✅ Setup GROQ_API_KEY in `.env`
2. ✅ Run `test_groq_fallback.py`
3. ✅ Test query với fallback
4. 📊 Monitor fallback rate: `result['stats']['used_fallback']`
5. 🔧 Tune model nếu cần (xem Groq Model Options)

---

## 🔗 Resources

- **Groq Console**: <https://console.groq.com>
- **API Docs**: <https://console.groq.com/docs/quickstart>
- **Models**: <https://console.groq.com/docs/models>
- **Pricing**: <https://groq.com/pricing>

---

**Date**: 2025-11-18  
**Version**: 1.0  
**Author**: RAG System Development Team
