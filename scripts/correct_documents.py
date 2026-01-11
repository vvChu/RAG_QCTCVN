"""
Auto-Correction Script for Document Classification

Scans all documents, detects misclass

ifications, and corrects:
- File naming
- Directory categorization
- Document registry metadata
"""
import os
import shutil
import json
from pathlib import Path
from document_analyzer import analyze_document

import unicodedata
import re

# Title mapping (Vietnamese → Vietnamese No Accents short titles)
TITLE_MAP = {
    "Số liệu điều kiện tự nhiên": "SoLieuDieuKienTuNhien",
    "Quy hoạch xây dựng": "QuyHoachXayDung",
    "Nhà chung cư": "NhaChungCu",
    "An toàn cháy": "AnToanChay",
    "Sửa đổi": "SuaDoi",
    "Nhà ở": "NhaO",
    "Luật xây dựng": "LuatXayDung",
    "DWELLINGS -BASIC PRINCIPLES FOR DESIGN": "NhaO_NguyenTacCoBan",
    "DWELLINGS": "NhaO",
    # Add more as needed
}

def remove_accents(input_str):
    """Remove accents from Vietnamese string."""
    if not input_str:
        return ""
    s1 = unicodedata.normalize('NFD', input_str)
    s2 = ''.join(c for c in s1 if unicodedata.category(c) != 'Mn')
    s3 = s2.replace('đ', 'd').replace('Đ', 'D')
    return s3

def get_short_title(vietnamese_title):
    """Convert Vietnamese title to short Vietnamese No Accents title."""
    if not vietnamese_title:
        return "TaiLieu"
    
    # Check map first
    for vn, no_accent in TITLE_MAP.items():
        if vn.lower() in vietnamese_title.lower():
            return no_accent
            
    # Clean prefixes
    prefixes = [
        "Quy chuẩn kỹ thuật quốc gia về",
        "Tiêu chuẩn quốc gia về",
        "Quy chuẩn kỹ thuật quốc gia",
        "Tiêu chuẩn quốc gia",
        "về "
    ]
    
    clean_title = vietnamese_title
    for prefix in prefixes:
        if clean_title.lower().startswith(prefix.lower()):
            clean_title = clean_title[len(prefix):].strip()
    
    # Default: Remove accents and take first 4 words
    clean_title = remove_accents(clean_title)
    # Remove special chars but keep spaces for splitting
    clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', clean_title)
    words = clean_title.split()[:6] # Increased to 6 to get more context
    return "".join([w.capitalize() for w in words])

def generate_correct_filename(metadata, original_date="2022-01-01"):
    """Generate correct filename from metadata."""
    std_type = metadata['standard_type']
    # Replace : with - and / with -
    std_num = metadata['standard_number'].replace(":", "-").replace("/", "-")
    
    # If standard number doesn't have suffix (e.g. /BXD), try to append it if known
    if std_type == "QCVN" and "BXD" not in std_num:
        std_num += "-BXD"
        
    raw_title = metadata.get('standard_title', '')
    print(f"    DEBUG: Raw Title: '{raw_title}'")
    title = get_short_title(raw_title)
    print(f"    DEBUG: Short Title: '{title}'")
    ext = metadata['file_ext']
    
    return f"{original_date}_{std_type}_{std_num}_{title}_v01{ext}"

def scan_and_correct(data_dir="data", dry_run=False):
    """
    Scan all documents and apply corrections.
    """
    print("="*60)
    print("INTELLIGENT DOCUMENT CLASSIFICATION & CORRECTION")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")
    
    corrections = []
    documents_dir = Path(data_dir) / "documents"
    
    # Scan all category directories
    for category_dir in documents_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        for file_path in category_dir.glob("*.*"):
            if file_path.suffix.lower() not in ['.pdf', '.docx']:
                continue
            
            print(f"Analyzing: {file_path.name}")
            
            # Analyze document
            metadata = analyze_document(file_path)
            
            if 'error' in metadata:
                print(f"  ⚠ Error: {metadata['error']}\n")
                continue
            
            # Determine correct classification
            if metadata['has_standard']:
                # Document contains a standard
                correct_category = metadata['standard_type'].lower()
                correct_category_dir = documents_dir / correct_category
                
                # Generate correct filename (preserve date from current filename if possible)
                date_match = file_path.name[:10] if file_path.name[:10].count('-') == 2 else "2022-01-01"
                correct_filename = generate_correct_filename(metadata, date_match)
                correct_path = correct_category_dir / correct_filename
                
                # Check if correction needed
                if file_path != correct_path:
                    corrections.append({
                        'current_path': file_path,
                        'correct_path': correct_path,
                        'metadata': metadata,
                        'reason': f"{metadata['standard_type']} {metadata['standard_number']}" + 
                                 (f" issued by {metadata['issuing_doc_type']} {metadata['issuing_doc_number']}" 
                                  if metadata['issuing_doc_type'] else "")
                    })
                    print(f"  ✗ NEEDS CORRECTION")
                    print(f"    Current: {category_dir.name}/{file_path.name}")
                    print(f"    Correct: {correct_category}/{correct_filename}")
                else:
                    print(f"  ✓ Already correct")
            else:
                print(f"  ✓ Standalone document (correctly placed)")
            
            print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total analyzed: {len(list(documents_dir.rglob('*.pdf'))) + len(list(documents_dir.rglob('*.docx')))}")
    print(f"Needs correction: {len(corrections)}")
    print()
    
    if corrections and not dry_run:
        print("Applying corrections...")
        for correction in corrections:
            current = correction['current_path']
            correct = correction['correct_path']
            
            # Ensure target directory exists
            correct.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(current), str(correct))
            print(f"  ✓ Moved: {current.name} → {correct.parent.name}/{correct.name}")
        
        print(f"\n✓ Applied {len(corrections)} corrections")
    
    elif corrections and dry_run:
        print(f"Would apply {len(corrections)} corrections (dry run)")
    
    return corrections

def update_metadata_registry(corrections, data_dir="data"):
    """Update document_registry.json with corrected info."""
    registry_path = Path(data_dir) / "metadata" / "document_registry.json"
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # Update each corrected document
    for correction in corrections:
        metadata = correction['metadata']
        correct_path = correction['correct_path']
        
        std_id = f"{metadata['standard_type']}_{metadata['standard_number'].replace(':', '-')}"
        
        # Find and update in registry
        for doc in registry['documents']:
            if doc['id'] == std_id or correct_path.name in doc.get('filename', ''):
                doc['filename'] = correct_path.name
                doc['category'] = metadata['standard_type']
                doc['directory'] = f"data/documents/{metadata['standard_type'].lower()}"
                
                # Add issuing info if present
                if metadata['issuing_doc_type']:
                    doc['issued_by'] = {
                        'type': metadata['issuing_doc_type'],
                        'number': metadata['issuing_doc_number']
                    }
                break
    
    # Save updated registry
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Updated metadata registry")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-correct document classification")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    
    args = parser.parse_args()
    
    corrections = scan_and_correct(args.data_dir, args.dry_run)
    
    if corrections and not args.dry_run:
        update_metadata_registry(corrections, args.data_dir)
        print("\n" + "="*60)
        print("✓ ALL CORRECTIONS APPLIED")
        print("="*60)

if __name__ == "__main__":
    main()
