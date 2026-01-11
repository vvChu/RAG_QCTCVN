# Data Strategy & SOTA RAG Improvements

## 1. File Naming Convention (Standardization)

To ensure efficient metadata extraction and organization, we propose the following naming convention for the `data/` directory.

### Format

`[YYYY-MM-DD]_[Type]_[Number]_[Description]_[Version].[ext]`

### Components

1. **Date (`YYYY-MM-DD`)**: Issuance date. Critical for "effective date" filtering.
2. **Type**: Document type abbreviation.
    - `QCVN`: Quy chuẩn kỹ thuật quốc gia (National Technical Regulation)
    - `TCVN`: Tiêu chuẩn quốc gia (National Standard)
    - `ND-CP`: Nghị định Chính phủ (Government Decree)
    - `TT-BXD`: Thông tư Bộ Xây dựng (Circular)
    - `LUAT`: Luật (Law)
3. **Number**: Official number (replace `/` with `-` to avoid path issues).
4. **Description**: Short, English or Vietnamese slug (no spaces, use underscores).
5. **Version** (Optional): `v1`, `v2` for updates.

### Examples

| Current Name | Proposed Name |
| :--- | :--- |
| `175_2024_ND-CP.pdf` | `2024-05-10_ND-CP_175-2024_ThiHanhLuatVienThong_v1.pdf` |
| `QCVN06_2022.docx` | `2022-11-30_QCVN_06-2022_AnToanChayNhaCongTrinh_v1.docx` |
| `SỬA ĐỔI 1_2023 QCVN 06_2022.docx` | `2023-10-16_QCVN_06-2022-SD1_SuaDoiAnToanChay_v1.docx` |

---

## 2. SOTA Legal RAG Strategy

### A. Advanced Chunking (Structure-Aware)

Legal documents are hierarchical trees, not flat text.

- **Current**: We use `StructuralChunker` (Chapter -> Article -> Clause). This is already close to SOTA.
- **Improvement**: **"Context-Enriched Chunking"**.
  - *Technique*: Prepend the "Parent Path" to every chunk.
  - *Example*: Instead of just "Clause 3: ...", the chunk text becomes:
        `Document: QCVN 06:2022 > Chapter 4: Fire Protection > Article 4.5 > Clause 3: ...`
  - *Benefit*: The embedding model understands the full context even if the clause text is generic.

### B. Indexing Strategy (Parent-Child)

- **Concept**: Decouple "Search Units" from "Generation Units".
- **Implementation**:
  - **Child Chunks (Small)**: Index individual Clauses (sentences) for high-precision vector search.
  - **Parent Chunks (Large)**: Store the full Article or Chapter text.
  - **Retrieval**: Search against Child Chunks, but return the Parent Chunk to the LLM.
  - *Benefit*: Precise matching + Full context for generation.

### C. Advanced Retrieval

1. **Hybrid Search (Implemented)**: Dense (Vector) + Sparse (BM25/Splade). Essential for legal terms (e.g., "PCCC").
2. **Cross-Encoder Reranking (Recommended)**:
    - Use a model like `bge-reranker-v2-m3`.
    - Re-score the top 50 results. This is the single most effective accuracy booster for RAG.
3. **Metadata Filtering**:
    - Use the **File Naming Convention** to auto-tag chunks with `Year`, `Type`, and `Authority`.
    - Allow users to filter: "Show me regulations from 2024 only".

### D. Specialized Embeddings

- **Current**: `BAAI/bge-m3` (General purpose, multilingual). Very good.
- **SOTA**: Fine-tuned Legal Embeddings (e.g., `voyage-law-2`, `saigon-nlp/Vietnamese-Legal-BERT`).
- **Recommendation**: Stick with `bge-m3` for now (it's excellent for Vietnamese), but experiment with `Legal-BERT` if accuracy stalls.

### E. Graph RAG (Future)

- Build a Knowledge Graph linking:
  - `[Document A]` --(amends)--> `[Document B]`
  - `[Concept: Fire Safety]` --(related to)--> `[Concept: High-rise]`
- Allows answering complex questions like "How has the fire safety regulation changed since 2020?".

---

## 3. Implementation Roadmap

1. **Immediate**: Rename files in `data/` to new convention.
2. **Short-term**: Implement **Context-Enriched Chunking** (prepend headers).
3. **Mid-term**: Add **Cross-Encoder Reranking**.
