# app/policy_chunker.py

import re
from typing import List, Optional, Tuple, Dict

from .models import PolicyChunk

SECTION_PATTERNS = [
    re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z\s&/()\-]{3,}$"),
    re.compile(r"^[A-Z][A-Z\s&/()\-]{3,}$"),
]

CLAUSE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$")


def _is_section_header(line: str) -> bool:
    return any(pat.match(line) for pat in SECTION_PATTERNS)


def _extract_clause(line: str) -> Optional[str]:
    match = CLAUSE_PATTERN.match(line)
    if match:
        return match.group(1)
    return None


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # avoid breaking on abbreviations or decimals
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _semantic_refine(text: str) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    refined: List[str] = []
    current: List[str] = []
    for sent in sentences:
        if not current:
            current.append(sent)
            continue

        # Start a new chunk on strong list-like boundaries or if chunk becomes too long.
        is_list_start = bool(re.match(r"^(\d+(?:\.\d+)*\)|\(|-)", sent))
        projected_len = len(" ".join(current)) + 1 + len(sent)
        if is_list_start or projected_len > 900:
            refined.append(" ".join(current).strip())
            current = [sent]
        else:
            current.append(sent)

    if current:
        refined.append(" ".join(current).strip())

    return [r for r in refined if r]


def _normalize_section(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _normalize_text(lines: List[str]) -> str:
    return " ".join([ln.strip() for ln in lines if ln.strip()]).strip()


def policy_aware_chunk(pages: List[Dict]) -> List[PolicyChunk]:
    chunks: List[PolicyChunk] = []
    current_section = "UNSPECIFIED"
    current_clause: Optional[str] = None
    page_context: Dict[int, Tuple[str, Optional[str]]] = {}

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        buffer_lines: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffer_lines, chunks
            if not buffer_lines:
                return
            raw_text = _normalize_text(buffer_lines)
            buffer_lines = []
            if not raw_text:
                return
            for piece in _semantic_refine(raw_text):
                if piece:
                    chunks.append(
                        PolicyChunk(
                            text=piece,
                            section=current_section,
                            clause=current_clause,
                            page=page_number,
                            content_type="text",
                        )
                    )

        for line in lines:
            if _is_section_header(line):
                flush_buffer()
                current_section = _normalize_section(line)
                current_clause = None
                continue

            clause = _extract_clause(line)
            if clause:
                flush_buffer()
                current_clause = clause
                buffer_lines.append(line)
                continue

            buffer_lines.append(line)

        flush_buffer()
        page_context[page_number] = (current_section, current_clause)

    # Table chunks (rows as independent chunks)
    for page in pages:
        page_number = page["page_number"]
        tables = page.get("tables", []) or []
        section, clause = page_context.get(page_number, ("UNSPECIFIED", None))
        for table in tables:
            if not table:
                continue
            for row in table:
                if not row:
                    continue
                cells = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                if not cells:
                    continue
                row_text = " | ".join(cells)
                chunks.append(
                    PolicyChunk(
                        text=row_text,
                        section=section,
                        clause=clause,
                        page=page_number,
                        content_type="table",
                    )
                )

    return chunks
