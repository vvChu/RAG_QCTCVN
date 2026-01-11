from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore()
store.load_collection()

# Query first 5 documents to see exact names
res = store.collection.query(
    expr='id != ""',
    output_fields=["document_name", "id"],
    limit=100
)

print(f"Total results: {len(res)}")
seen_names = set()
for r in res:
    name = r.get("document_name")
    if name not in seen_names:
        print(f"Stored Name: '{name}'")
        seen_names.add(name)

# Specifically check one of the failed files
target = "QCVN_10_2025_PCCC_Trang_bi_bo_tri_phuong_tien_PCCC.pdf"
print(f"\nChecking target: '{target}'")
res_target = store.collection.query(
    expr=f'document_name == "{target}"',
    output_fields=["id"]
)
print(f"Exact match count: {len(res_target)}")

# Check partial match
res_partial = store.collection.query(
    expr=f'document_name like "%QCVN_10%"',
    output_fields=["document_name"],
    limit=1
)
if res_partial:
    print(f"Partial match found name: '{res_partial[0].get('document_name')}'")
