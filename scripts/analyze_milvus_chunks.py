"""
Analyze actual chunks in Milvus to determine optimal chunk size.
"""
import sys
sys.path.insert(0, 'src')

from ccba_rag.core.settings import settings
from pymilvus import MilvusClient

# Connect
client = MilvusClient(
    uri=f'https://{settings.milvus_host}:{settings.milvus_port}',
    token=f'{settings.milvus_user}:{settings.milvus_password}'
)

# Query sample chunks
results = client.query(
    collection_name=settings.milvus_collection_name,
    filter='',
    output_fields=['text', 'document_name', 'article', 'clause'],
    limit=500
)

# Analyze lengths
lengths = [len(r.get('text', '')) for r in results]

print(f"Sample size: {len(lengths)} chunks")
print(f"Min length: {min(lengths)} chars")
print(f"Max length: {max(lengths)} chars")
print(f"Average: {sum(lengths)/len(lengths):.0f} chars")
print(f"Median: {sorted(lengths)[len(lengths)//2]} chars")

# Distribution
print("\nLength Distribution:")
brackets = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
for b in brackets:
    count = sum(1 for l in lengths if l <= b)
    pct = count / len(lengths) * 100
    print(f"  <= {b:4d} chars: {count:3d}/{len(lengths)} ({pct:5.1f}%)")

# Percentiles
sorted_lens = sorted(lengths)
p50 = sorted_lens[int(len(sorted_lens) * 0.50)]
p75 = sorted_lens[int(len(sorted_lens) * 0.75)]
p90 = sorted_lens[int(len(sorted_lens) * 0.90)]
p95 = sorted_lens[int(len(sorted_lens) * 0.95)]
p99 = sorted_lens[int(len(sorted_lens) * 0.99)]

print("\nPercentiles:")
print(f"  50th: {p50:,} chars")
print(f"  75th: {p75:,} chars")
print(f"  90th: {p90:,} chars")
print(f"  95th: {p95:,} chars")
print(f"  99th: {p99:,} chars")

# Sample by document
print("\nChunks by Document:")
doc_counts = {}
for r in results:
    doc = r.get('document_name', 'Unknown')
    if doc not in doc_counts:
        doc_counts[doc] = []
    doc_counts[doc].append(len(r.get('text', '')))

for doc, lens in sorted(doc_counts.items(), key=lambda x: -len(x[1]))[:10]:
    avg = sum(lens) / len(lens)
    print(f"  {doc[:40]:40s}: {len(lens):3d} chunks, avg {avg:.0f} chars")

# Recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print(f"\nCurrent MAX_CHUNK_CHARS: 2000")
print(f"Observed p95: {p95:,} chars")

if p95 <= 2000:
    print(f"\n✅ Current setting (2000) is GOOD - covers 95% of chunks")
else:
    print(f"\n⚠️ Consider increasing to {p95:,} to cover 95% of chunks")
