"""
Comprehensive Document Verification

Verifies that all document content is properly extracted and chunked:
1. Page-by-page tracking - which pages have content
2. Table extraction check - ensure tables are captured
3. Image/diagram logging - count non-text elements
4. Coverage calculation - % of content chunked
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re

sys.path.insert(0, 'src')


@dataclass
class PageStats:
    """Statistics for a single page."""
    page_number: int
    char_count: int
    has_text: bool
    table_count: int
    image_count: int


@dataclass
class DocumentVerification:
    """Complete verification result for a document."""
    file_name: str
    total_pages: int
    total_chars: int
    pages_with_text: int
    pages_empty: int
    table_count: int
    image_count: int
    chunks_created: int
    chunk_chars: int
    coverage_percent: float
    page_stats: List[PageStats]
    issues: List[str]


def extract_pdf_with_details(pdf_path: str) -> Tuple[str, List[PageStats]]:
    """Extract PDF with detailed page-by-page stats."""
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        full_text = []
        page_stats = []
        
        for page_num, page in enumerate(doc, 1):
            # Extract text
            text = page.get_text("text")
            
            # Count tables (approximate by looking for table-like patterns)
            tables = len(re.findall(r'(?:\|.*\|)|(?:\+[-+]+\+)', text))
            
            # Count images
            images = len(page.get_images())
            
            stats = PageStats(
                page_number=page_num,
                char_count=len(text),
                has_text=len(text.strip()) > 50,
                table_count=tables,
                image_count=images
            )
            page_stats.append(stats)
            full_text.append(text)
        
        doc.close()
        return "\n".join(full_text), page_stats
    
    except Exception as e:
        print(f"  Error: {e}")
        return "", []


def get_indexed_chunks(document_name: str) -> List[Dict]:
    """Get indexed chunks from Milvus."""
    try:
        from ccba_rag.retrieval.vectorstores.milvus import MilvusStore
        
        store = MilvusStore()
        store.load_collection()
        
        # Use simple count query since we just need the count
        results = store.collection.query(
            expr=f'document_name == "{document_name}"',
            output_fields=['text'], # We need text to sum chars
            limit=5000 # Increase limit to capture all chunks
        )
        
        return results
    except Exception as e:
        print(f"  Milvus error: {e}")
        return []


def verify_document(file_path: Path) -> DocumentVerification:
    """Perform comprehensive verification of a document."""
    print(f"\n{'='*60}")
    print(f"Verifying: {file_path.name}")
    
    # Extract with details
    text, page_stats = extract_pdf_with_details(str(file_path))
    
    if not page_stats:
        return DocumentVerification(
            file_name=file_path.name,
            total_pages=0,
            total_chars=0,
            pages_with_text=0,
            pages_empty=0,
            table_count=0,
            image_count=0,
            chunks_created=0,
            chunk_chars=0,
            coverage_percent=0,
            page_stats=[],
            issues=["Could not extract PDF"]
        )
    
    # Calculate stats
    total_pages = len(page_stats)
    total_chars = sum(p.char_count for p in page_stats)
    pages_with_text = sum(1 for p in page_stats if p.has_text)
    pages_empty = total_pages - pages_with_text
    table_count = sum(p.table_count for p in page_stats)
    image_count = sum(p.image_count for p in page_stats)
    
    print(f"  Pages: {total_pages}")
    print(f"  Characters: {total_chars:,}")
    print(f"  Pages with text: {pages_with_text}")
    print(f"  Empty pages: {pages_empty}")
    print(f"  Tables detected: {table_count}")
    print(f"  Images detected: {image_count}")
    
    # Get indexed chunks
    chunks = get_indexed_chunks(file_path.name)
    chunks_created = len(chunks)
    chunk_chars = sum(len(c.get('text', '')) for c in chunks)
    
    print(f"  Chunks indexed: {chunks_created}")
    
    # Calculate coverage
    coverage = (chunk_chars / total_chars * 100) if total_chars > 0 else 0
    print(f"  Coverage: {coverage:.1f}%")
    
    # Identify issues
    issues = []
    
    if pages_empty > 0:
        issues.append(f"[WARN] {pages_empty} empty pages (may be scanned)")
    
    if coverage < 50:
        issues.append(f"[ERROR] Low coverage: {coverage:.1f}%")
    
    if image_count > 0 and chunks_created == 0:
        issues.append(f"[WARN] {image_count} images but no text extracted")
    
    if total_chars < 100 and total_pages > 0:
        issues.append("[WARN] Very little text (likely scanned PDF)")
    
    # Page-by-page issues
    for p in page_stats:
        if p.image_count > 0 and not p.has_text:
            issues.append(f"  Page {p.page_number}: has images but no text")
            
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  [OK] No issues detected")
    
    return DocumentVerification(
        file_name=file_path.name,
        total_pages=total_pages,
        total_chars=total_chars,
        pages_with_text=pages_with_text,
        pages_empty=pages_empty,
        table_count=table_count,
        image_count=image_count,
        chunks_created=chunks_created,
        chunk_chars=chunk_chars,
        coverage_percent=coverage,
        page_stats=page_stats,
        issues=issues
    )


def main():
    data_dir = Path("data/documents")
    
    # Find PDFs
    pdf_files = list(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    results = []
    for pdf_path in pdf_files[:15]:  # Limit for speed
        result = verify_document(pdf_path)
        results.append(result)
    
    # Summary Report
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_pages = sum(r.total_pages for r in results)
    total_images = sum(r.image_count for r in results)
    total_tables = sum(r.table_count for r in results)
    total_chunks = sum(r.chunks_created for r in results)
    
    files_with_issues = [r for r in results if r.issues]
    
    print(f"\n📊 Statistics:")
    print(f"   Total files: {len(results)}")
    print(f"   Total pages: {total_pages}")
    print(f"   Total images: {total_images}")
    print(f"   Total tables: {total_tables}")
    print(f"   Total chunks: {total_chunks}")
    
    print(f"\n⚠️ Files with issues: {len(files_with_issues)}")
    for r in files_with_issues:
        print(f"   - {r.file_name}")
        for issue in r.issues[:3]:  # Show first 3 issues
            print(f"     {issue}")
    
    # Coverage summary
    print("\n📈 Coverage Summary:")
    high_coverage = [r for r in results if r.coverage_percent >= 90]
    medium_coverage = [r for r in results if 50 <= r.coverage_percent < 90]
    low_coverage = [r for r in results if r.coverage_percent < 50]
    
    print(f"   ✅ High (≥90%): {len(high_coverage)} files")
    print(f"   ⚠️ Medium (50-90%): {len(medium_coverage)} files")
    print(f"   ❌ Low (<50%): {len(low_coverage)} files")
    
    if low_coverage:
        print("\n   Files needing attention:")
        for r in low_coverage:
            print(f"     - {r.file_name}: {r.coverage_percent:.1f}%")


if __name__ == "__main__":
    main()
