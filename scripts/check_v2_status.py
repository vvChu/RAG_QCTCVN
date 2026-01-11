import sys
import os

# Add src to path so ccba_rag can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore(collection_name="legal_docs_v2")

if store.has_collection():
    store.load_collection()
    count = store.collection.num_entities
    print(f"Entities in 'legal_docs_v2': {count}")
else:
    print("'legal_docs_v2' does not exist yet.")
