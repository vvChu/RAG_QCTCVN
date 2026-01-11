# Quick Start Guide

## Bước 1: Setup môi trường

### 1.1. Clone và cài đặt

```bash
git clone <your-repo-url>
cd RAG_QCTCVN

python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc .\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 1.2. Start Milvus

```bash
docker-compose up -d

# Kiểm tra Milvus đang chạy
docker ps
```

### 1.3. Configure API Keys

```bash
cp .env.example .env
nano .env  # Chỉnh GEMINI_API_KEY
```

## Bước 2: Chuẩn bị dữ liệu

```bash
# Tạo thư mục và copy PDF files
mkdir -p data/pdfs
cp /path/to/your/pdfs/*.pdf data/pdfs/
```

## Bước 3: Index documents

```bash
python src/main.py --mode index --pdf-dir data/pdfs
```

**Expected output:**
```
Initializing RAG System...
Loading BGE-M3 model: BAAI/bge-m3
Model loaded successfully!
...
INDEXING COMPLETED: 245 chunks indexed
```

## Bước 4: Query

```bash
python src/main.py --mode query --question "Chiều cao tối thiểu của tầng 1?"
```

## Bước 5: Python Script

Tạo file `test.py`:

```python
from src.main import RAGSystem

system = RAGSystem()

# Query
result = system.query("Khoảng cách an toàn giữa công trình?")

print("Answer:", result['answer'])
```

Chạy:

```bash
python test.py
```

## Troubleshooting

### Lỗi: Không kết nối được Milvus

```bash
docker-compose restart
docker logs milvus-standalone
```

### Lỗi: GPU Out of Memory

Thêm vào `.env`:

```env
BGE_USE_FP16=False
```

Hoặc sử dụng CPU:

```python
system = RAGSystem()
system.embedder.model.device = 'cpu'
```

### Lỗi: GEMINI_API_KEY not found

Đảm bảo file `.env` có:

```env
GEMINI_API_KEY=your_actual_api_key
```

## Next Steps

- Xem [README.md](../README.md) để hiểu kiến trúc
- Xem [SPECIFICATION.md](SPECIFICATION.md) cho chi tiết kỹ thuật
- Chạy [examples/evaluation_example.py](../examples/evaluation_example.py) để đánh giá
