from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore()
store.load_collection()

# Query specifically for QCVN_10
expr = 'document_name like "%QCVN_10%"'
res = store.collection.query(
    expr=expr,
    output_fields=["id"]
)
print(f"QCVN_10 chunks: {len(res)}")

# Query for one of the batch files too
expr2 = 'document_name like "%LUAT_62_2020%"'
res2 = store.collection.query(
    expr=expr2,
    output_fields=["id"]
)
print(f"LUAT_62 chunks: {len(res2)}")

# Query for ND 06
expr3 = 'document_name like "%ND_06_2021%"'
res3 = store.collection.query(
    expr=expr3,
    output_fields=["id"]
)
print(f"ND_06 chunks: {len(res3)}")
