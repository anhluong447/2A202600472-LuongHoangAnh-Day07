from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        current = ""
        for part in parts:
            if part in [". ", "! ", "? ", ".\n"]:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        if sep == "":
            parts = list(current_text)
        else:
            parts = current_text.split(sep)

        final_chunks = []
        current_chunk = ""

        for part in parts:
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                final_chunks.extend(self._split(part, remaining_separators[1:]))
            else:
                separator_to_use = sep if current_chunk else ""
                if len(current_chunk) + len(separator_to_use) + len(part) <= self.chunk_size:
                    current_chunk = current_chunk + separator_to_use + part
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk)
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # cosine_similarity = dot(a, b) / (||a|| * ||b||)
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        
        # FixedSizeChunker
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        results['fixed_size'] = {
            'count': len(fixed_chunks),
            'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            'chunks': fixed_chunks
        }
        
        # SentenceChunker
        sentence_chunks = SentenceChunker().chunk(text)
        results['by_sentences'] = {
            'count': len(sentence_chunks),
            'avg_length': sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
            'chunks': sentence_chunks
        }
        
        # RecursiveChunker
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)
        results['recursive'] = {
            'count': len(recursive_chunks),
            'avg_length': sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
            'chunks': recursive_chunks
        }
        
        return results

class MarkdownHeaderChunker:
    """
    Strategy 1: Chunk by Markdown Headers and inject metadata.
    Splits text by markdown headings and returns Document objects with
    page_title and section_title metadata injected.
    """
    
    def chunk_document(self, doc: Document) -> list[Document]:
        from .models import Document
        lines = doc.content.split('\n')
        chunks = []
        current_h1 = ""
        current_h2 = ""
        current_content = []
        
        def push_chunk():
            text = "\n".join(current_content).strip()
            if text:
                meta = dict(doc.metadata)
                if current_h1: meta["page_title"] = current_h1
                if current_h2: meta["section_title"] = current_h2
                
                if len(text) > 1500:
                    rc = RecursiveChunker(chunk_size=800)
                    sub_texts = rc.chunk(text)
                    for i, stext in enumerate(sub_texts):
                        new_doc = Document(id=f"{doc.id}_{len(chunks)}_{i}", content=stext, metadata=meta)
                        chunks.append(new_doc)
                else:
                    new_doc = Document(id=f"{doc.id}_{len(chunks)}", content=text, metadata=meta)
                    chunks.append(new_doc)
            current_content.clear()
            
        for line in lines:
            if line.startswith("# "):
                push_chunk()
                current_h1 = line[2:].strip()
                current_content.append(line)
            elif line.startswith("## ") or line.startswith("### "):
                push_chunk()
                current_h2 = line.lstrip("#").strip()
                current_content.append(line)
            else:
                current_content.append(line)
                
        push_chunk()
        return chunks


class StructureAwareMarkdownChunker:
    """
    Chunk markdown text by header boundaries, returning plain strings.

    Used by openai_eval.py.  Falls back to RecursiveChunker when a
    section exceeds *chunk_size* characters.
    """

    def __init__(self, chunk_size: int = 450) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        lines = text.split("\n")
        chunks: list[str] = []
        current_content: list[str] = []

        def push():
            block = "\n".join(current_content).strip()
            if not block:
                current_content.clear()
                return
            if len(block) <= self.chunk_size:
                chunks.append(block)
            else:
                rc = RecursiveChunker(chunk_size=self.chunk_size)
                chunks.extend(rc.chunk(block))
            current_content.clear()

        for line in lines:
            if re.match(r"^#{1,3}\s", line):
                push()
            current_content.append(line)

        push()
        return chunks
