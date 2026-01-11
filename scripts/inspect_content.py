from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore()
store.load_collection()

target = "LUAT_62_2020_PHAPLY_sua_doi_bo_sung_mot_so_dieu_cua_luat_xay_dung.pdf"
print(f"Inspecting content for: {target}")

res = store.collection.query(
    expr=f'document_name == "{target}"',
    output_fields=["text", "page_number"],
    limit=5
)

for i, r in enumerate(res):
    print(f"\n--- Chunk {i+1} (Page {r.get('page_number')}) ---")
    print(r.get("text")[:500]) # First 500 chars
