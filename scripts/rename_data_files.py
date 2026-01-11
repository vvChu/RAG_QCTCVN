"""
File Renaming Script for Data Directory Optimization

This script renames files to follow the standardized naming convention:
{YYYY-MM-DD}_{Category}_{DocNum}_{ShortTitle}_v{Version}.{ext}
"""
import os
import shutil
import json
from datetime import datetime

# Mapping of current files to new standardized names
RENAME_MAP = {
    # Current name -> New name
    "02_2022_TT-BXD_530884.pdf": "2022-01-01_TT-BXD_02-2022_Drainage_v1.pdf",
    "2012-01-01_TCVN_4451-2012_NhaO_NguyenTacCoBan_v1.pdf": "2012-01-01_TCVN_4451-2012_ResidentialBasics_v1.pdf",
    "2021-05-19_QCVN_01-2021_QuyHoachXayDung_v1.pdf": "2021-05-19_QCVN_01-2021_UrbanPlanning_v1.pdf",
    "2021-05-19_QCVN_04-2021_NhaChungCu_v1.pdf": "2021-05-19_QCVN_04-2021_Apartments_v1.pdf",
    "2022-11-30_QCVN_06-2022_AnToanChayNhaCongTrinh_v1.docx": "2022-11-30_QCVN_06-2022_FireSafety_v1.docx",
    "2023-10-16_QCVN_06-2022-SD1_SuaDoiAnToanChay_v1.docx": "2023-10-16_QCVN_06-2022-SD1_FireSafetyAmendment_v1.docx",
    "2024-05-10_ND-CP_175-2024_ThiHanhLuatXayDung_QuanLyHoatDongXD_v1.pdf": "2024-05-10_ND-CP_175-2024_ConstructionLaw_v1.pdf"
}

# Category directories
CATEGORY_DIRS = {
    "QCVN": "data/documents/qcvn",
    "TCVN": "data/documents/tcvn",
    "ND-CP": "data/documents/nd-cp",
    "TT-BXD": "data/documents/tt-bxd"
}

def get_category_from_filename(filename):
    """Extract category from filename."""
    if "QCVN" in filename:
        return "QCVN"
    elif "TCVN" in filename:
        return "TCVN"
    elif "ND-CP" in filename:
        return "ND-CP"
    elif "TT-BXD" in filename:
        return "TT-BXD"
    return None

def rename_and_organize(data_dir="data", dry_run=False):
    """Rename files and organize into category directories."""
    
    print("="*60)
    print("FILE RENAMING AND ORGANIZATION")
    print("="*60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will rename files)'}")
    print()
    
    renamed_count = 0
    moved_count = 0
    
    for old_name, new_name in RENAME_MAP.items():
        old_path = os.path.join(data_dir, old_name)
        
        if not os.path.exists(old_path):
            print(f"⚠ SKIP: {old_name} (file not found)")
            continue
        
        # Get category and target directory
        category = get_category_from_filename(new_name)
        if category:
            target_dir = CATEGORY_DIRS[category]
            new_path = os.path.join(target_dir, new_name)
        else:
            new_path = os.path.join(data_dir, new_name)
        
        print(f"📄 {old_name}")
        print(f"   → {new_name}")
        if category:
            print(f"   📁 Category: {category}")
        print()
        
        if not dry_run:
            # Ensure target directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Move and rename file
            shutil.move(old_path, new_path)
            renamed_count += 1
            if category:
                moved_count += 1
    
    # Move test data
    test_file = os.path.join(data_dir, "gold_standard_qa.json")
    if os.path.exists(test_file):
        test_target = os.path.join(data_dir, "test", "gold_standard_qa.json")
        print(f"📊 gold_standard_qa.json")
        print(f"   → test/gold_standard_qa.json")
        print()
        
        if not dry_run:
            os.makedirs(os.path.dirname(test_target), exist_ok=True)
            shutil.move(test_file, test_target)
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    if dry_run:
        print(f"Would rename: {len(RENAME_MAP)} files")
        print(f"Would organize: {moved_count} files by category")
    else:
        print(f"✓ Renamed: {renamed_count} files")
        print(f"✓ Organized: {moved_count} files by category")
        print(f"✓ Test data moved to test/")
    print("="*60)
    
    return renamed_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rename and organize data files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    
    args = parser.parse_args()
    
    rename_and_organize(args.data_dir, args.dry_run)
