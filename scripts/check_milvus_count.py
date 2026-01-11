from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore()
store.load_collection()

num = store.collection.num_entities
print(f"Total entities: {num}")

# Query specifically for QCVN_01
res = store.collection.query(
    expr='document_name like "QCVN_01%"',
    output_fields=["count(*)"]
)
print(f"QCVN_01 chunks: {len(res)}")
