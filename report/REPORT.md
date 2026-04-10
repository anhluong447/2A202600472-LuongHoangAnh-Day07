# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Lương Hoàng Anh]
**Nhóm:** [Nhom 64]
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (gần 1.0) nghĩa là hai vector embedding có hướng gần giống nhau trong không gian nhiều chiều, tức là hai đoạn văn bản mang ý nghĩa ngữ nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Học phí VinUni là bao nhiêu mỗi năm?"
- Sentence B: "Chi phí học tập hàng năm tại VinUniversity là bao nhiêu?"
- Tại sao tương đồng: Cả hai câu đều hỏi về cùng một chủ đề (chi phí đại học), dùng từ vựng đồng nghĩa ("học phí" ↔ "chi phí học tập"), cùng nhắm vào entity VinUni.

**Ví dụ LOW similarity:**
- Sentence A: "Học phí VinUni là bao nhiêu mỗi năm?"
- Sentence B: "Thời tiết hôm nay ở Hà Nội ra sao?"
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (tài chính giáo dục vs thời tiết), không chia sẻ ngữ nghĩa nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector, không phụ thuộc vào độ dài (magnitude) của vector. Điều này quan trọng vì hai đoạn văn bản dài ngắn khác nhau vẫn có thể mang cùng ý nghĩa — cosine sẽ cho điểm cao, trong khi Euclidean distance sẽ bị ảnh hưởng bởi sự khác biệt magnitude.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:* step = chunk_size - overlap = 500 - 50 = 450
> Số chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = **23 chunks**
> (Chunk cuối có thể ngắn hơn 500 ký tự)

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> step = 500 - 100 = 400, số chunks = ceil(9500 / 400) + 1 = 24 + 1 = **25 chunks** (tăng từ 23 lên 25). Overlap nhiều hơn giúp bảo toàn ngữ cảnh giữa các chunk — một câu bị cắt ở cuối chunk trước sẽ được lặp lại ở đầu chunk sau, tránh mất thông tin khi retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tuyển sinh, Đào tạo, và Dịch vụ sinh viên của Đại học VinUni.

**Tại sao nhóm chọn domain này?**
> Domain này có cấu trúc rõ ràng, chứa nhiều thông tin chi tiết (FAQs, yêu cầu tuyển sinh, học phí, v.v.). Đây là một use-case rất thực tế cho RAG trong hệ thống chatbot tư vấn thông tin sinh viên, đòi hỏi sự chính xác cao để không cung cấp sai lệch thông tin quy chế. Ngoài ra, tài liệu được viết bằng Markdown với heading rõ ràng, rất phù hợp để thử nghiệm nhiều chiến lược chunking khác nhau.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | vinuni_01_why_vinuni.md | vinuni.edu.vn | ~2.3k | source_file, source_url |
| 2 | vinuni_02_academics_dao-tao.md | vinuni.edu.vn | ~2.9k | source_file, source_url |
| 3 | vinuni_03_admission_tuyen-sinh-faq.md | admissions.vinuni.edu.vn | ~4.8k | source_file, source_url |
| 4 | vinuni_04_admission_hoc-phi-hoc-bong.md | admissions.vinuni.edu.vn | ~4.9k | source_file, source_url |
| 5 | vinuni_05_admission_chuong-trinh-dao-tao.md | admissions.vinuni.edu.vn | ~3.5k | source_file, source_url |
| 6 | vinuni_06_campus_services.md | vinuni.edu.vn | ~3.3k | source_file, source_url |
| 7 | vinuni_07_registrar_tot-nghiep.md | registrar.vinuni.edu.vn | ~2.8k | source_file, source_url |
| 8 | vinuni_08_student_portal.md | vinuni.edu.vn | ~1.8k | source_file, source_url |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source_file` | string | `vinuni_03_admission_tuyen-sinh-faq.md` | Giúp truy xuất ngược lại nguồn tham chiếu, lọc theo từng bộ tài liệu. |
| `source_url` | string | `https://admissions.vinuni.edu.vn/...` | Cho phép đối chiếu kết quả retrieval với expected_sources trong benchmark. |
| `page_title` | string | `Why VinUni` | Cung cấp bối cảnh cấp cao để LLM hiểu đoạn chunk thuộc topic lớn nào. |
| `section_title` | string | `Học bổng` | Phân mảnh các FAQ chính xác theo từng section, tăng độ chính xác semantic search. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| vinuni_03 | FixedSizeChunker (`fixed_size`) | 15 | ~200 | Thường cắt ngang câu ngẫu nhiên, phá vỡ bullet list |
| vinuni_03 | SentenceChunker (`by_sentences`) | 20 | ~150 | Tốt hơn, nhưng các bullet list dễ bị tách rời khỏi heading |
| vinuni_03 | RecursiveChunker (`recursive`) | 12 | ~250 | Rất tốt trong đa số văn bản, giữ nguyên đoạn |
| vinuni_06 | FixedSizeChunker (`fixed_size`) | 10 | ~200 | Cắt rời thông tin hotline khỏi tên dịch vụ |
| vinuni_06 | SentenceChunker (`by_sentences`) | 14 | ~140 | Tách bullet points nhưng giữ câu hoàn chỉnh |
| vinuni_06 | RecursiveChunker (`recursive`) | 8 | ~260 | Giữ context khá tốt nhờ ưu tiên `\n\n` |

### Strategy Của Tôi

**Loại:** custom strategy (`StructureAwareMarkdownChunker` / `MarkdownHeaderChunker`)

**Mô tả cách hoạt động:**
> Thuật toán duyệt từng dòng của file Markdown, mỗi khi gặp heading (`#`, `##`, `###`) thì gom toàn bộ đoạn text trước đó thành 1 chunk duy nhất. Các heading title được inject và gắn trực tiếp làm metadata `page_title` và `section_title` cho các Document đó. Nếu có section nào vượt quá `chunk_size` (mặc định 450 chars) thì dùng `RecursiveChunker` làm fallback để chia nhỏ hơn mà vẫn giữ ngữ cảnh. Cách tiếp cận này đảm bảo mỗi chunk chứa trọn vẹn một đơn vị ngữ nghĩa (section).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu VinUni hoàn toàn dùng định dạng Markdown có cấu trúc đề mục rõ ràng: mỗi chủ đề (Học phí, Điều kiện tốt nghiệp, KTX...) nằm gọn dưới một heading. Gom chunk theo Header giúp cô lập các đoạn Q&A / FAQs hoàn hảo thay vì bị phá vỡ vì giới hạn số chữ. Metadata `section_title` tự động sinh ra cũng giúp LLM hiểu ngữ cảnh tốt hơn.

**Code snippet (nếu custom):**
```python
class StructureAwareMarkdownChunker:
    def __init__(self, chunk_size: int = 450):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks, current_content = [], []

        def push():
            block = "\n".join(current_content).strip()
            if not block:
                current_content.clear(); return
            if len(block) <= self.chunk_size:
                chunks.append(block)
            else:
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(block))
            current_content.clear()

        for line in lines:
            if re.match(r"^#{1,3}\s", line):
                push()
            current_content.append(line)
        push()
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| vinuni_03 | best baseline (Recursive) | 12 | ~250 | Khá, nhưng khó lấy được Q&A trọn vẹn |
| vinuni_03 | **của tôi (MarkdownHeader)** | 9 | ~350 | Rất tốt, trích xuất chuẩn từng section |
| vinuni_06 | best baseline (Recursive) | 8 | ~260 | Đôi khi trộn lẫn thông tin 2 dịch vụ khác nhau |
| vinuni_06 | **của tôi (MarkdownHeader)** | 7 | ~300 | Mỗi dịch vụ (Y tế, Tâm lý, KTX) nằm gọn 1 chunk |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Recall@3 | MRR@5 | Gold Fact Coverage | Điểm mạnh | Điểm yếu |
|-----------|----------|----------|-------|-------|-----------|----------|
| **Tôi** | StructureAwareMarkdownChunker | **1.00** | **0.867** | 0.340 | Hoàn hảo chia chunk theo heading, recall tuyệt đối | Gold fact coverage bị hạ bởi cross-language matching |
| [Teammate] | [Strategy khác] | **1.00** | **0.900** | **0.540** | MRR rất cao, fact coverage tốt hơn | - |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Cả hai strategy đều đạt Recall@3 = 1.0, chứng tỏ dữ liệu VinUni có cấu trúc tốt giúp mọi chiến lược đều tìm đúng tài liệu. Tuy nhiên, teammate đạt MRR@5 = 0.90 và Gold Fact Coverage = 0.54 cao hơn tôi, cho thấy chiến lược của teammate xếp hạng chunk chính xác hơn ở một số query. `StructureAwareMarkdownChunker` của tôi vẫn có lợi thế rõ ràng về tính nhất quán ngữ nghĩa (mỗi chunk = 1 section trọn vẹn), phù hợp nhất cho domain có cấu trúc Markdown chuẩn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `r'(\. |! |? |.\n)'` để phát hiện ranh giới câu dựa trên dấu chấm/chấm than/hỏi theo sau bởi khoảng trắng hoặc newline. Sau khi tách thành danh sách câu, gom mỗi `max_sentences_per_chunk` câu thành 1 chunk. Edge case: nếu câu cuối không kết thúc bằng dấu chấm, vẫn được gom vào chunk cuối cùng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy thử tách text bằng separator đầu tiên trong danh sách ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Nếu một phần tách ra vẫn quá dài, gọi đệ quy `_split` với separator tiếp theo. Base case: text đã ngắn hơn `chunk_size` thì trả về nguyên bản; hoặc hết separator thì cắt cứng theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: Gọi `embedding_fn` trên nội dung mỗi Document để tạo vector, lưu cả record (id, content, metadata, embedding) vào `self._store` (list in-memory). Nếu ChromaDB có sẵn, đồng thời ghi vào collection. `search`: Embed query thành vector, tính dot product với tất cả vector đã lưu, sort giảm dần theo score, trả top-k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: **Lọc trước** (pre-filter) danh sách records theo metadata_filter, sau đó mới chạy similarity search trên tập đã lọc. Nếu không có filter thì fallback về search thường. `delete_document`: Duyệt toàn bộ `self._store`, loại bỏ records có id trùng hoặc metadata[doc_id] trùng, trả True nếu có xóa được ít nhất 1 record.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` để lấy top-k chunks liên quan, nối nội dung các chunks thành chuỗi context. Xây prompt theo format `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` rồi truyền vào `llm_fn`. LLM sẽ dựa trên context được inject để tạo câu trả lời grounded.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Học phí VinUni mỗi năm là bao nhiêu?" | "Chi phí học tập hàng năm tại VinUniversity?" | high | 0.82 | ✅ |
| 2 | "Điều kiện tốt nghiệp cử nhân?" | "GPA tối thiểu để ra trường?" | high | 0.71 | ✅ |
| 3 | "Sinh viên năm nhất phải ở KTX không?" | "Thời tiết hôm nay thế nào?" | low | 0.08 | ✅ |
| 4 | "VinUni hợp tác với Cornell University" | "Cornell là đối tác chiến lược của VinUni" | high | 0.78 | ✅ |
| 5 | "Học bổng toàn phần cho sinh viên xuất sắc" | "Ngân hàng cho vay mua nhà lãi suất thấp" | low | 0.15 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 có actual score = 0.71, thấp hơn expected. Mặc dù cả hai câu đều hỏi về tốt nghiệp, embedding model xử lý "điều kiện tốt nghiệp" và "GPA tối thiểu" như hai khái niệm có liên quan nhưng không đồng nhất. Điều này cho thấy embeddings biểu diễn ngữ nghĩa chi tiết hơn ta nghĩ — chúng phân biệt được "điều kiện tổng thể" vs "một tiêu chí cụ thể" (GPA), không chỉ đơn thuần gom nhóm theo chủ đề.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân bằng `openai_eval.py` (Embedding: `all-MiniLM-L6-v2`, LLM: `gpt-4.1-mini`).

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Điều kiện tiếng Anh, GPA khuyến nghị và 3 kỳ tuyển sinh đại học của VinUni là gì? | IELTS 6.5+, TOEFL iBT 79+, PTE 58+, CAE 176+; GPA 8.0+; 3 kỳ: 15/10-15/01, 15/02-15/05, 15/06-15/08 |
| 2 | Học phí niêm yết mỗi năm của Điều dưỡng và các ngành khác là bao nhiêu? Sau hỗ trợ 35% còn khoảng bao nhiêu? | Điều dưỡng ~349.650.000 VND, ngành khác ~815.850.000 VND. Sau giảm 35%: ~227 triệu và ~530 triệu. |
| 3 | VinUni cho phép những hình thức học liên ngành thế nào, và có thể học rút ngắn không? | Ngành đơn, kép, chính-phụ, song bằng, bằng tích hợp; Tín chỉ nên có thể rút ngắn còn 3.5 năm. |
| 4 | Điều kiện tốt nghiệp cử nhân tại VinUni là gì? GPA tối thiểu bao nhiêu? | Hoàn thành tín chỉ, VinCore, tiếng Anh, môn bắt buộc, xử lý điểm I, GPA ≥ 2.00/4.00, không vi phạm. |
| 5 | Sinh viên năm nhất có bắt buộc ở ký túc xá không? Dịch vụ y tế và tham vấn tâm lý nằm ở đâu, hotline nào? | Bắt buộc KTX năm 1. Y tế phòng I119 hotline (+84) 866 200 019. Tâm lý phòng I118 hotline (+84) 868 900 016. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Doc | Recall@3 | MRR@5 | Gold Fact Coverage | Agent Answer (tóm tắt) |
|---|-------|---------------------|----------|-------|------|------------------------|
| 1 | Tiếng Anh/GPA/3 kỳ | vinuni_03 ✅ | 1.0 | 1.0 | 0.625 | Liệt kê đủ IELTS 6.5+, TOEFL 79, PTE 58, CAE 176, GPA 8.0+. Thiếu 3 mốc ngày cụ thể. |
| 2 | Học phí | vinuni_04 ✅ | 1.0 | 1.0 | 0.600 | Trả đúng 349.650.000 và 815.850.000 VND, giảm 35% còn 227tr và 530tr. |
| 3 | Liên ngành | vinuni_02 (partial) | 1.0 | 0.333 | 0.333 | Liệt kê 5 hình thức + rút ngắn 3.5 năm. Chunk đúng (vinuni_05) rớt xuống hạng 3. |
| 4 | Tốt nghiệp | vinuni_07 ✅ | 1.0 | 1.0 | 0.143 | Liệt kê đủ 7 điều kiện + GPA 2.00/4.00. Coverage thấp do gold_facts viết tiếng Anh. |
| 5 | KTX & y tế | vinuni_06 ✅ | 1.0 | 1.0 | 0.000 | Đúng KTX bắt buộc, nhưng thiếu hotline y tế/tâm lý chi tiết (hallucinate số 18008189). |

### Các Metrics IR Tiêu Chuẩn (Average)

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **Recall@3** | **1.00** | 100% query đều kéo đúng doc nguồn vào Top-3 |
| **Recall@5** | **1.00** | 100% query có doc đúng trong Top-5 |
| **MRR@5** | **0.867** | Chunk đúng thường nằm Top-1. Riêng q3 rớt hạng 3 |
| **Gold Fact Coverage** | **0.340** | Thấp do cross-language (gold_facts EN ↔ answer VI) |

### So sánh với Thành viên khác

| Metric | Tôi (MarkdownHeader) | Teammate |
|--------|---------------------|----------|
| Recall@3 | **1.00** | **1.00** |
| Recall@5 | **1.00** | **1.00** |
| MRR@5 | 0.867 | **0.900** |
| Gold Fact Coverage | 0.340 | **0.540** |

### Failure Analysis

**Query 3 (Liên ngành) — MRR thấp:** Chunk đúng (`vinuni_05_admission_chuong-trinh-dao-tao`) rớt xuống hạng 3 vì file `vinuni_02_academics_dao-tao` cũng đề cập đến "đào tạo" và "liên ngành" → embedding model không phân biệt rõ admission FAQ vs academic overview.

**Query 5 (KTX/Y tế) — Gold Fact Coverage = 0:** Chunk top-1 chỉ chứa tổng quan Campus Services, không chứa chi tiết phòng số I118/I119 và hotline cụ thể. LLM không tìm thấy thông tin trong context → hallucinate số hotline sai (18008189 là hotline tuyển sinh, không phải y tế). **Giải pháp:** tăng `top_k` từ 3 lên 15 (như trong `openai_eval.py`) hoặc cải thiện chunking để tách riêng từng dịch vụ.

**Gold Fact Coverage tổng thể thấp (0.34 vs teammate 0.54):** Chủ yếu do phương pháp đánh giá substring-based. `gold_facts` viết bằng tiếng Anh nhưng GPT trả lời bằng tiếng Việt. Ví dụ: gold fact = `"complete minimum credits within program duration"` nhưng answer = `"Hoàn thành số tín chỉ tối thiểu"` → không match được. Teammate có thể đạt cao hơn nhờ strategy cho phép GPT trả lời sát format gold hơn hoặc top_k lớn hơn.

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

### Tự Đánh Giá Kết Quả Retrieval

**1. Retrieval Precision**
- Top-3 có chứa chunk thật sự liên quan không? → **Có, Recall@3 = 1.00** trên toàn bộ 5 query.
- Score có tách được kết quả tốt và nhiễu không? → Chunk đúng thường có score cao nhất (0.65–0.70), chunk nhiễu thấp hơn rõ rệt (~0.55).

**2. Chunk Coherence**
- Chunk có giữ được ý trọn vẹn không? → **Có**, nhờ chia chunk theo Markdown heading, mỗi chunk = 1 section trọn vẹn.
- Strategy nào làm chunk dễ đọc và dễ retrieve hơn? → `StructureAwareMarkdownChunker` vượt trội so với FixedSize vì không cắt ngang câu hay bullet list.

**3. Metadata Utility**
- `search_with_filter()` có giúp tăng độ chính xác không? → Có thể lọc theo `source_file` hoặc `section_title` để thu hẹp phạm vi tìm kiếm.
- Filter có quá chặt, làm mất kết quả tốt không? → Trong bài này chưa kích hoạt filter vì bộ tài liệu nhỏ (8 file), nhưng metadata sẵn sàng cho scale lớn hơn.

**4. Grounding Quality**
- Câu trả lời của agent có thật sự dựa trên retrieved context không? → q1-q4: **Có**, GPT trả lời sát context. q5: **Không hoàn toàn** — GPT hallucinate hotline sai (18008189) vì context thiếu thông tin chi tiết.
- Có thể chỉ ra chunk nào hỗ trợ câu trả lời không? → **Có**, mỗi query đều log rõ top-5 retrieved chunks với doc_id và score.

**5. Data Strategy Impact**
- Bộ tài liệu nhóm chọn có phù hợp với benchmark queries không? → **Phù hợp**, 5 query đều có tài liệu tương ứng trong data/.
- Strategy chunking / metadata của bạn có hợp với domain không? → **Rất hợp**, domain VinUni dùng Markdown có heading rõ ràng → MarkdownHeader chunking khai thác tối đa cấu trúc này.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Teammate đạt MRR@5 = 0.90 (cao hơn tôi 0.867), cho thấy việc tinh chỉnh tham số chunk_size và overlap có thể cải thiện thứ hạng kết quả đáng kể. Một bài học quan trọng: không chỉ cần tìm đúng document mà còn phải đẩy chunk chính xác nhất lên vị trí đầu tiên.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Data quality quan trọng hơn model selection. Cùng một model embedding (all-MiniLM-L6-v2), nhóm nào có tài liệu cấu trúc tốt hơn và metadata schema phù hợp hơn sẽ cho kết quả retrieval vượt trội. Chunking strategy phải được thiết kế dựa trên đặc điểm cấu trúc của domain cụ thể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ (1) tách riêng mỗi dịch vụ trong Campus Services thành chunk riêng biệt để tránh mất thông tin chi tiết hotline/phòng số (failure case q5); (2) thêm metadata `topic_category` (admissions/academics/services) để enable metadata filtering khi query; và (3) tăng `top_k` lên 10-15 để cung cấp nhiều context hơn cho LLM, giảm nguy cơ hallucination.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **90 / 100** |
