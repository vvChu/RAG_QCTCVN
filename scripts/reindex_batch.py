import subprocess
import sys
import time

files = [
    "data/documents/03_PHAP_LY_DU_AN/ND_06_2021_QLDA_Quan_ly_chat_luong_thi_cong_bao_tri_cong_trinh.pdf",
    "data/documents/03_PHAP_LY_DU_AN/TT_174_2021_QLDA_huong_dan_nd_06_2021_quan_ly_chat_luong_cong_trinh.pdf",
    "data/documents/03_PHAP_LY_DU_AN/TT_24_2025_QLDA_sua_doi_tt_174_2021_tt_bqp.pdf"
]

print(f"Starting batch re-index for {len(files)} files...")

for i, file in enumerate(files, 1):
    print(f"\n[{i}/{len(files)}] Indexing: {file}")
    try:
        cmd = [sys.executable, "scripts/fix_missing_files.py", file]
        # Set PYTHONPATH
        env = {"PYTHONPATH": "src"}
        
        # Increased timeout for robustness (10 minutes per file)
        # subprocess.run doesn't have timeout by default unless specified, but capturing output is good.
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS]:\n{result.stdout}")
        else:
            print(f"[FAILED] (Exit {result.returncode}):\n{result.stderr}")
            print(f"Stdout: {result.stdout}")
            
    except Exception as e:
        print(f"[ERROR] running script: {e}")
    
    # Small pause to let resources release
    time.sleep(5)

print("\nBatch processing complete.")
