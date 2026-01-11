"""
Document Structure Analysis for Optimal Chunk Size

Analyzes documents in data/documents to determine:
1. Average Article (Điều) length
2. Average Clause (Khoản) length  
3. Token distribution
4. Recommended chunk size
"""

import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, 'src')

# Regex patterns for Vietnamese legal structure
CHAPTER_PATTERN = re.compile(r'(CHƯƠNG\s+[IVXLCDM\d]+[.:]\s*[^\n]+)', re.IGNORECASE | re.MULTILINE)
ARTICLE_PATTERN = re.compile(r'(Điều\s+\d+[a-z]?\.?\s*[^\n]*)', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'(\d+\.\s+[^\n]+)', re.MULTILINE)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX."""
    try:
        import docx
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def analyze_document(file_path: Path) -> dict:
    """Analyze a single document."""
    print(f"\nAnalyzing: {file_path.name}")
    
    # Extract text
    if file_path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(str(file_path))
    elif file_path.suffix.lower() in ['.docx', '.doc']:
        text = extract_text_from_docx(str(file_path))
    else:
        return None
    
    if not text or len(text) < 100:
        print(f"  Warning: No/little text extracted")
        return None
    
    # Find articles (Điều)
    articles = ARTICLE_PATTERN.findall(text)
    
    # Find all article positions
    article_positions = [m.start() for m in ARTICLE_PATTERN.finditer(text)]
    
    # Calculate lengths between articles
    article_lengths = []
    for i in range(len(article_positions)):
        start = article_positions[i]
        end = article_positions[i + 1] if i + 1 < len(article_positions) else len(text)
        length = end - start
        if length < 10000:  # Filter out anomalies
            article_lengths.append(length)
    
    # Find clauses (Khoản)
    clauses = CLAUSE_PATTERN.findall(text)
    clause_lengths = [len(c) for c in clauses if len(c) < 5000]
    
    stats = {
        'file': file_path.name,
        'total_chars': len(text),
        'num_articles': len(articles),
        'num_clauses': len(clauses),
        'article_lengths': article_lengths,
        'clause_lengths': clause_lengths,
        'avg_article_length': sum(article_lengths) / len(article_lengths) if article_lengths else 0,
        'avg_clause_length': sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0,
        'max_article_length': max(article_lengths) if article_lengths else 0,
        'max_clause_length': max(clause_lengths) if clause_lengths else 0,
    }
    
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Articles (Điều): {stats['num_articles']}")
    print(f"  Avg article length: {stats['avg_article_length']:.0f} chars")
    print(f"  Max article length: {stats['max_article_length']:.0f} chars")
    print(f"  Clauses (Khoản): {stats['num_clauses']}")
    print(f"  Avg clause length: {stats['avg_clause_length']:.0f} chars")
    
    return stats

def main():
    data_dir = Path("data/documents")
    
    if not data_dir.exists():
        print("data/documents not found!")
        return
    
    # Find all documents
    pdf_files = list(data_dir.rglob("*.pdf"))
    docx_files = list(data_dir.rglob("*.docx"))
    all_files = pdf_files + docx_files
    
    print(f"Found {len(all_files)} documents ({len(pdf_files)} PDFs, {len(docx_files)} DOCX)")
    print("=" * 60)
    
    all_stats = []
    all_article_lengths = []
    all_clause_lengths = []
    
    for file_path in all_files[:10]:  # Limit to first 10 for speed
        stats = analyze_document(file_path)
        if stats:
            all_stats.append(stats)
            all_article_lengths.extend(stats['article_lengths'])
            all_clause_lengths.extend(stats['clause_lengths'])
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    if all_article_lengths:
        print(f"\nArticle (Điều) Length Distribution:")
        print(f"  Total articles analyzed: {len(all_article_lengths)}")
        print(f"  Min: {min(all_article_lengths):,} chars")
        print(f"  Max: {max(all_article_lengths):,} chars")
        print(f"  Average: {sum(all_article_lengths)/len(all_article_lengths):,.0f} chars")
        print(f"  Median: {sorted(all_article_lengths)[len(all_article_lengths)//2]:,} chars")
        
        # Percentiles
        sorted_lens = sorted(all_article_lengths)
        p75 = sorted_lens[int(len(sorted_lens) * 0.75)]
        p90 = sorted_lens[int(len(sorted_lens) * 0.90)]
        p95 = sorted_lens[int(len(sorted_lens) * 0.95)]
        print(f"  75th percentile: {p75:,} chars")
        print(f"  90th percentile: {p90:,} chars")
        print(f"  95th percentile: {p95:,} chars")
    
    if all_clause_lengths:
        print(f"\nClause (Khoản) Length Distribution:")
        print(f"  Total clauses analyzed: {len(all_clause_lengths)}")
        print(f"  Average: {sum(all_clause_lengths)/len(all_clause_lengths):,.0f} chars")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if all_article_lengths:
        avg_article = sum(all_article_lengths) / len(all_article_lengths)
        median_article = sorted(all_article_lengths)[len(all_article_lengths)//2]
        
        # Recommend chunk size that captures 90% of articles
        recommended = sorted_lens[int(len(sorted_lens) * 0.90)]
        
        print(f"\n✅ Recommended MAX_CHUNK_CHARS: {recommended:,}")
        print(f"   (Covers 90% of articles without truncation)")
        print(f"\n   Alternative options:")
        print(f"   - Conservative (75%): {p75:,} chars")
        print(f"   - Aggressive (95%): {p95:,} chars")
        
        # Token estimate
        tokens_estimate = recommended * 0.35  # Vietnamese ~0.35 tokens/char
        print(f"\n   Token estimate: ~{tokens_estimate:.0f} tokens")
        if tokens_estimate > 1024:
            print(f"   ⚠️ May need BGE_MAX_LENGTH > 1024")
        else:
            print(f"   ✅ Fits within BGE_MAX_LENGTH=1024")

if __name__ == "__main__":
    main()
