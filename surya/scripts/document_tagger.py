"""Document Metadata Tagger for Surya OCR.
Extracts document classification, titles, entities, summaries, and block metadata.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from surya.recognition.schema import PageOCRResult


def _strip_html(html_str: str) -> str:
    """Strip basic HTML tags."""
    if not html_str:
        return ""
    clean = re.sub(r"<[^>]+>", " ", html_str)
    return re.sub(r"\s+", " ", clean).strip()


def extract_key_entities(text: str) -> List[str]:
    """Extract dates, numbers, monetary amounts, and key codes using regex patterns."""
    entities = []

    # Dates (e.g. 2026-09-02, 02/09/2026, September 2, 2026)
    date_pattern = r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b"
    dates = re.findall(date_pattern, text, flags=re.IGNORECASE)
    entities.extend([f"📅 Date: {d}" for d in set(dates[:4])])

    # Monetary amounts (e.g. $10,000, ₹50,000, USD 500, EUR 1,200)
    money_pattern = r"\b(?:[\$\₹\€\£]\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|INR|EUR|GBP|RS|RUPEES))\b"
    money = re.findall(money_pattern, text, flags=re.IGNORECASE)
    entities.extend([f"💰 Amount: {m}" for m in set(money[:4])])

    # Key Codes / Identifiers (e.g. ID-1234, INV-990, REF: 8829)
    code_pattern = r"\b(?:INV|ID|REF|NO|CODE|DOC|TATA)[-:\s]*[A-Z0-9]{3,12}\b"
    codes = re.findall(code_pattern, text, flags=re.IGNORECASE)
    entities.extend([f"🔑 ID: {c}" for c in set(codes[:4])])

    return entities[:8]


def classify_document_type(page: PageOCRResult, full_text: str) -> str:
    """Classify the document type based on layout blocks and key vocabulary."""
    text_lower = full_text.lower()

    if any(k in text_lower for k in ("invoice", "bill to", "payment due", "total amount", "tax invoice", "subtotal")):
        return "Invoice / Billing Document"
    elif any(k in text_lower for k in ("specification", "technical specification", "architecture", "diagram", "schematic")):
        return "Technical Specification"
    elif any(k in text_lower for k in ("report", "financial statement", "quarterly", "balance sheet", "annual report")):
        return "Financial & Corporate Report"
    elif any(k in text_lower for k in ("agreement", "contract", "terms and conditions", "memorandum", "party")):
        return "Legal Agreement / Contract"
    elif any(k in text_lower for k in ("form", "application", "declaration", "field")):
        return "Form / Application Sheet"
    elif any(b.label == "Table" for b in page.blocks) and len(page.blocks) <= 5:
        return "Tabular Statement"
    elif any(b.label == "Picture" for b in page.blocks) and len(page.blocks) <= 4:
        return "Diagrammatic / Visual Document"
    else:
        return "General Document"


def tag_document_page(
    page: PageOCRResult,
    pil_image: Image.Image,
    source_name: str,
    page_num: int = 1,
) -> Dict[str, Any]:
    """Generates rich document metadata tags from Surya PageOCRResult."""
    blocks = page.blocks
    
    # Extract Titles and Headers
    titles = [_strip_html(b.html) for b in blocks if b.label == "Title" and b.html.strip()]
    doc_title = titles[0] if titles else f"{source_name} (Page {page_num})"

    # Full Page Raw Text
    text_parts = [_strip_html(b.html) for b in blocks if b.html.strip()]
    full_text = " ".join(text_parts)

    # Classify & Extract Entities
    doc_type = classify_document_type(page, full_text)
    entities = extract_key_entities(full_text)

    # Count Layout Types
    layout_counts: Dict[str, int] = {}
    for b in blocks:
        lbl = b.label or "Text"
        layout_counts[lbl] = layout_counts.get(lbl, 0) + 1

    # Extract Section Headings for Summary
    headers = [_strip_html(b.html) for b in blocks if b.label in ("SectionHeader", "Header") and b.html.strip()]
    summary_parts = headers[:3] if headers else text_parts[:2]
    summary = " | ".join(summary_parts[:3]) if summary_parts else "Document page extracted."

    return {
        "source_name": source_name,
        "page_num": page_num,
        "document_title": doc_title,
        "document_type": doc_type,
        "summary": summary,
        "entities": entities,
        "layout_counts": layout_counts,
        "total_blocks": len(blocks),
        "total_text_length": len(full_text),
    }
