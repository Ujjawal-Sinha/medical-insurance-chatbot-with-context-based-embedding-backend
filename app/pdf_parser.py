# app/pdf_parser.py

import io
from typing import List, Dict, Any

import pdfplumber


def parse_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            except Exception:
                text = ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            pages.append({
                "page_number": idx,
                "text": text,
                "tables": tables,
            })
    return pages
