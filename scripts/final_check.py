from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
from ccba_rag.utils.logging import configure_logging

configure_logging()
store = MilvusStore()
store.load_collection()

files_to_check = [
    "QCVN_10_2025_PCCC_Trang_bi_bo_tri_phuong_tien_PCCC.pdf",
    "LUAT_62_2020_PHAPLY_sua_doi_bo_sung_mot_so_dieu_cua_luat_xay_dung.pdf",
    "ND_06_2021_QLDA_Quan_ly_chat_luong_thi_cong_bao_tri_cong_trinh.pdf",
    "TT_174_2021_QLDA_huong_dan_nd_06_2021_quan_ly_chat_luong_cong_trinh.pdf",
    "TT_24_2025_QLDA_sua_doi_tt_174_2021_tt_bqp.pdf"
]

print("Final Chunk Counts:")
for fname in files_to_check:
    res = store.collection.query(
        expr=f'document_name == "{fname}"',
        output_fields=["id"]
    )
    print(f"{fname}: {len(res)} chunks")
