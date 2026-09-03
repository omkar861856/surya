"""Surya2 streamlit app — exercise layout, recognition, table_rec via the
inference manager. Detection + OCR-error stay in their own torch paths."""

from __future__ import annotations

import base64
import io
import re
import tempfile
import time
from typing import List

import pypdfium2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image
from surya.detection import TextDetectionResult
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import PageOCRResult
from surya.settings import settings
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult

from surya.scripts.doc_exporter import create_docx_from_surya_page, create_docx_from_surya_pages, create_docx_from_markdown
from surya.scripts.document_tagger import tag_document_page


# KaTeX & Document Layout HTML wrapper.
_KATEX_HEAD = r"""<!doctype html><html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
html,body{background:#f8f9fa; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; margin:0; padding:16px;}
.document-paper{background:#ffffff; max-width:900px; margin:0 auto; padding:32px 40px; border-radius:8px; box-shadow:0 4px 20px rgba(0,0,0,0.08); border:1px solid #e9ecef;}
table{border-collapse:collapse; margin:14px 0; width:100%;} td,th{border:1px solid #ced4da; padding:8px 12px; color:#111111;}
th{background-color:#f1f4f8; font-weight:600; color:#003366;}
tr:nth-child(even){background-color:#f8f9fa;}
[data-label="Title"]{font-size:26px; font-weight:700; color:#003366; margin-top:16px; margin-bottom:12px; border-bottom:2px solid #003366; padding-bottom:8px;}
[data-label="SectionHeader"]{font-size:20px; font-weight:600; color:#005599; margin-top:20px; margin-bottom:8px;}
[data-label="PageHeader"],[data-label="Header"]{font-size:12px; color:#6c757d; text-transform:uppercase; border-bottom:1px solid #dee2e6; padding-bottom:4px; margin-bottom:16px;}
[data-label="PageFooter"],[data-label="Footer"]{font-size:12px; color:#6c757d; border-top:1px solid #dee2e6; padding-top:8px; margin-top:24px;}
.ocr-block{margin-bottom:12px;}
.ocr-image-block img{max-width:100%; height:auto; border-radius:6px; border:1px solid #dee2e6; box-shadow:0 4px 12px rgba(0,0,0,0.08);}
</style></head><body>
"""

_KATEX_TAIL = r"""
<script>
renderMathInElement(document.body, {
  delimiters: [
    {left: "\\[", right: "\\]", display: true},
    {left: "\\(", right: "\\)", display: false}
  ],
  throwOnError: false
});
</script></body></html>
"""

_MATH_RE = re.compile(r"<math\b([^>]*)>(.*?)</math>", re.DOTALL | re.IGNORECASE)


def _math_to_katex(html_str: str) -> str:
    """Rewrite <math>...</math> tags into KaTeX \\( \\) / \\[ \\] delimiters."""

    def repl(m: "re.Match") -> str:
        attrs, inner = m.group(1), m.group(2)
        if re.search(r"""display\s*=\s*["']block["']""", attrs):
            return "\\[" + inner + "\\]"
        return "\\(" + inner + "\\)"

    return _MATH_RE.sub(repl, html_str or "")


def render_ocr_html(html_str: str, height: int = 400) -> None:
    """Render OCR HTML with math typeset by KaTeX (iframe component)."""
    components.html(
        _KATEX_HEAD + _math_to_katex(html_str) + _KATEX_TAIL,
        height=height,
        scrolling=True,
    )


def _crop_to_b64(pil_img: Image.Image, bbox: list[float], pad: int = 4) -> str:
    """Crop a bounding box region from a PIL image and return base64 encoded PNG."""
    x0 = max(0, int(bbox[0]) - pad)
    y0 = max(0, int(bbox[1]) - pad)
    x1 = min(pil_img.size[0], int(bbox[2]) + pad)
    y1 = min(pil_img.size[1], int(bbox[3]) + pad)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = pil_img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _assemble_page_html(page: PageOCRResult, pil_image: Image.Image | None = None) -> str:
    """Reconstruct a styled paper HTML document from a PageOCRResult,
    embedding cropped images for visual blocks (pictures, figures, diagrams, logos)."""
    parts: List[str] = ['<div class="document-paper">']
    img_w, img_h = pil_image.size if pil_image else (800, 1000)

    for blk in page.blocks:
        x0, y0, x1, y1 = (int(c) for c in blk.bbox)
        label = blk.label or "Text"
        body = blk.html or ""

        # Check if block is a visual block or skipped OCR block
        is_visual = (
            blk.skipped
            or label in ("Picture", "Figure", "Image", "Diagram", "Logo", "Stamp", "Photo")
            or not body.strip()
        )

        if is_visual and pil_image:
            b64_img = _crop_to_b64(pil_image, blk.bbox)
            if b64_img:
                parts.append(
                    f'<div class="ocr-block ocr-image-block" data-bbox="{x0} {y0} {x1} {y1}" data-label="{label}" style="text-align: center; margin: 16px 0;">'
                    f'<img src="data:image/png;base64,{b64_img}" alt="{label}" />'
                    f'<div style="font-size: 11px; color: #6c757d; margin-top: 4px;">📷 Extracted Image ({label})</div>'
                    f'</div>'
                )
            continue

        if body.strip():
            parts.append(
                f'<div class="ocr-block" data-bbox="{x0} {y0} {x1} {y1}" data-label="{label}">{body}</div>'
            )

    parts.append('</div>')
    return "\n".join(parts)


def _assemble_spatial_page_html(
    page: PageOCRResult, pil_image: Image.Image, canvas_width: int = 850
) -> str:
    """Reconstruct an exact 1:1 spatial positioning replica of the original document layout
    using exact mathematical pixel scaling coordinates.
    """
    img_w, img_h = pil_image.size
    if img_w <= 0 or img_h <= 0:
        return ""

    scale = canvas_width / float(img_w)
    canvas_height = int(img_h * scale)

    parts: List[str] = [
        f'<div class="spatial-page-wrapper" style="width: 100%; display: flex; justify-content: center; background: #323639; padding: 24px 0; border-radius: 8px; margin-bottom: 24px;">',
        f'<div class="spatial-page-canvas" style="position: relative; width: {canvas_width}px; height: {canvas_height}px; background: #ffffff; box-shadow: 0 6px 24px rgba(0,0,0,0.3); border-radius: 4px; overflow: hidden;">',
    ]

    for blk in page.blocks:
        x0, y0, x1, y1 = (int(c) for c in blk.bbox)
        label = blk.label or "Text"
        body = blk.html or ""

        b_left = int(x0 * scale)
        b_top = int(y0 * scale)
        b_width = max(1, int((x1 - x0) * scale))
        b_height = max(1, int((y1 - y0) * scale))

        is_visual = (
            blk.skipped
            or label in ("Picture", "Figure", "Image", "Diagram", "Logo", "Stamp", "Photo")
            or not body.strip()
        )

        style_pos = (
            f"position: absolute; left: {b_left}px; top: {b_top}px; "
            f"width: {b_width}px; height: {b_height}px; "
            f"box-sizing: border-box; overflow: hidden; padding: 2px;"
        )

        if is_visual:
            b64_img = _crop_to_b64(pil_image, blk.bbox)
            if b64_img:
                parts.append(
                    f'<div style="{style_pos} z-index: 2;">'
                    f'<img src="data:image/png;base64,{b64_img}" style="width: 100%; height: 100%; object-fit: contain; display: block;" alt="{label}" />'
                    f'</div>'
                )
        elif body.strip():
            parts.append(
                f'<div style="{style_pos} font-size: 12px; line-height: 1.25; color: #111111; z-index: 1;">'
                f'{body}'
                f'</div>'
            )

    parts.append("</div></div>")
    return "\n".join(parts)


def _show_timing(label: str, elapsed_s: float, extra: str = "") -> None:
    """Render a small caption with wall-clock + optional extra detail."""
    detail = f" — {extra}" if extra else ""
    st.caption(f"⏱ {label}: {elapsed_s * 1000:.0f} ms ({elapsed_s:.2f}s){detail}")


@st.cache_resource()
def load_predictors_cached():
    manager = SuryaInferenceManager()
    layout_predictor = LayoutPredictor(manager)
    rec_predictor = RecognitionPredictor(manager)
    table_rec_predictor = TableRecPredictor(manager)

    # Lazy-import detection / ocr_error to keep startup snappy when the user
    # only wants VLM modes
    from surya.detection import DetectionPredictor
    from surya.ocr_error import OCRErrorPredictor

    return {
        "manager": manager,
        "layout": layout_predictor,
        "recognition": rec_predictor,
        "table_rec": table_rec_predictor,
        "detection": DetectionPredictor(),
        "ocr_error": OCRErrorPredictor(),
    }


@st.cache_resource()
def load_fast_layout():
    from surya.fast_layout import FastLayoutPredictor

    return FastLayoutPredictor()


def _layout_predictor(use_fast: bool):
    return load_fast_layout() if use_fast else predictors["layout"]


def text_detection(img) -> tuple[Image.Image, TextDetectionResult, float]:
    t = time.perf_counter()
    text_pred = predictors["detection"]([img])[0]
    elapsed = time.perf_counter() - t
    text_polygons = [p.polygon for p in text_pred.bboxes]
    det_img = draw_polys_on_image(text_polygons, img.copy())
    return det_img, text_pred, elapsed


def layout_detection(
    img, use_fast: bool = False
) -> tuple[Image.Image, LayoutResult, float]:
    t = time.perf_counter()
    pred = _layout_predictor(use_fast)([img])[0]
    elapsed = time.perf_counter() - t
    polygons = [p.polygon for p in pred.bboxes]
    labels = [
        f"{p.label}-{p.position}-c{p.count}-{round(p.confidence or 0, 2)}"
        for p in pred.bboxes
    ]
    annotated = draw_polys_on_image(
        polygons, img.copy(), labels=labels, label_font_size=14
    )
    return annotated, pred, elapsed


def block_ocr(img) -> tuple[Image.Image, PageOCRResult, LayoutResult, float, float]:
    """Layout → block crops → BLOCK_PROMPT. Returns layout + block-OCR timings."""
    t_layout = time.perf_counter()
    layout = predictors["layout"]([img])[0]
    layout_elapsed = time.perf_counter() - t_layout

    t_blocks = time.perf_counter()
    page_results = predictors["recognition"]([img], [layout])
    blocks_elapsed = time.perf_counter() - t_blocks
    page = page_results[0]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, layout, layout_elapsed, blocks_elapsed


def full_page_ocr(img) -> tuple[Image.Image, PageOCRResult, float]:
    """Single HIGH_ACCURACY_BBOX_PROMPT call on the whole page."""
    t = time.perf_counter()
    page_results = predictors["recognition"]([img], full_page=True)
    elapsed = time.perf_counter() - t
    page = page_results[0]
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)
    return annotated, page, elapsed


def table_recognition(
    img: Image.Image,
    mode: str,
    skip_table_detection: bool,
    use_fast_layout: bool = False,
) -> tuple[Image.Image, List[TableResult], float, float]:
    """Returns (annotated_img, table_preds, layout_elapsed, table_rec_elapsed)."""
    layout_elapsed = 0.0
    if skip_table_detection:
        table_imgs = [img]
        table_bboxes = [(0, 0, img.size[0], img.size[1])]
    else:
        t = time.perf_counter()
        layout = _layout_predictor(use_fast_layout)([img])[0]
        layout_elapsed = time.perf_counter() - t
        tables = [b for b in layout.bboxes if b.label in ("Table", "TableOfContents")]
        if not tables:
            return img.copy(), [], layout_elapsed, 0.0
        table_bboxes = [tuple(int(c) for c in b.bbox) for b in tables]
        table_imgs = [img.crop(b) for b in table_bboxes]

    t = time.perf_counter()
    if mode == "full":
        table_preds = predictors["table_rec"].predict_full(table_imgs)
    else:
        table_preds = predictors["table_rec"].predict_simple(table_imgs)
    table_rec_elapsed = time.perf_counter() - t

    out_img = img.copy()
    for pred, table_img, tbbox in zip(table_preds, table_imgs, table_bboxes):
        if pred.error or pred.mode != "simple" or not pred.rows:
            continue
        row_bboxes = [r.bbox for r in pred.rows]
        col_bboxes = [c.bbox for c in pred.cols]
        row_labels = [r.label for r in pred.rows]
        col_labels = [c.label for c in pred.cols]
        annot = table_img.copy()
        annot = draw_bboxes_on_image(
            row_bboxes, annot, labels=row_labels, label_font_size=14, color="blue"
        )
        annot = draw_bboxes_on_image(
            col_bboxes, annot, labels=col_labels, label_font_size=14, color="red"
        )
        # Paste annotated crop back at the table's position in the page.
        out_img.paste(annot, (tbbox[0], tbbox[1]))
    return out_img, table_preds, layout_elapsed, table_rec_elapsed


def ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)

        page_middle = page_count // 2
        page_range = range(
            max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count)
        )
        text = plain_text_output(f.name, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i : i + sample_len])

    results = predictors["ocr_error"](samples)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > 0.2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    try:
        page = doc[page_num - 1]
        png_image = page.render(scale=dpi / 72).to_pil().convert("RGB")
        return png_image
    finally:
        doc.close()


@st.cache_data()
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


st.set_page_config(layout="wide")
col1, col2 = st.columns([0.55, 0.45])

predictors = load_predictors_cached()

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.markdown(
        """
# Welcome Tata Power Team! ⚡

Welcome to the **Tata Power Document Intelligence & OCR Hub**.

We are delighted to bring you this advanced OCR and document analysis platform designed to effortlessly extract text, analyze document layouts, recognize tables, and digitize your documents with high speed and precision.

👈 **Get Started:** Upload a PDF document or image using the sidebar menu on the left to begin processing.
"""
    )
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    scan_scope = st.sidebar.radio(
        "📄 Document Scan Scope:",
        options=["🔄 Scan All Pages (Full Document - Default)", "📄 Single Page Only"],
        index=0,
        help="By default, all pages of the PDF are scanned and extracted sequentially.",
    )
    if scan_scope == "📄 Single Page Only":
        page_number = st.sidebar.number_input(
            f"Select page number (out of {page_count}):", min_value=1, value=1, max_value=page_count
        )
    else:
        page_number = 1  # Preview page index
    pil_image = get_page_image(in_file, page_number, settings.IMAGE_DPI_HIGHRES)
else:
    scan_scope = "Single Page Only"
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

run_unified_pipeline = st.sidebar.button("⚡ Run Unified Ideal Pipeline (Document Agnostic)", type="primary")
st.sidebar.markdown("---")
st.sidebar.caption("Individual Modular Features:")
run_full_page_ocr = st.sidebar.button("Run Full-Page OCR")
run_text_det = st.sidebar.button("Run Text Detection")
run_layout = st.sidebar.button("Run Layout Analysis")
run_table_rec = st.sidebar.button("Run Table Rec")
run_block_ocr = st.sidebar.button("Run Block OCR")
run_ocr_errors = st.sidebar.button("Run bad-PDF-text detection")

use_fast_layout = st.sidebar.checkbox(
    "Fast layout",
    value=True,
    help="Use the fast layout detector.",
)
table_mode = st.sidebar.radio(
    "Table mode",
    options=["simple", "full"],
    index=0,
    help="simple: rows+cols only. full: full HTML.",
)
skip_table_detection = st.sidebar.checkbox(
    "Skip table detection",
    value=False,
    help="Treat the entire page/image as a single table.",
)

if pil_image is None:
    st.stop()


if run_unified_pipeline:
    with col1:
        st.subheader("⚡ Unified Document-Agnostic Intelligence Pipeline")
        
        is_multi_page = (
            "pdf" in filetype
            and scan_scope == "🔄 Scan All Pages (Full Document - Default)"
            and page_count is not None
            and page_count > 1
        )

        source_name = in_file.name if hasattr(in_file, "name") else "Document"

        if is_multi_page:
            st.info(f"📚 **Full Document Mode**: Iteratively scanning and extracting all {page_count} pages...")
            prog_bar = st.progress(0, text="Starting document extraction...")
            
            all_pages_data: List[tuple[PageOCRResult, Image.Image]] = []
            all_html_parts: List[str] = []
            all_spatial_parts: List[str] = []
            all_annotated: List[tuple[int, Image.Image]] = []
            total_elapsed = 0
            doc_tags = {}

            for p_idx in range(1, page_count + 1):
                prog_bar.progress(
                    int((p_idx - 1) / page_count * 100),
                    text=f"Scanning Page {p_idx} of {page_count}..."
                )
                p_img = get_page_image(in_file, p_idx, settings.IMAGE_DPI_HIGHRES)
                ann_img, p_page, p_layout, p_ltime, p_btime = block_ocr(p_img)
                total_elapsed += (p_ltime + p_btime)
                all_pages_data.append((p_page, p_img))
                all_annotated.append((p_idx, ann_img))

                if p_idx == 1:
                    doc_tags = tag_document_page(p_page, p_img, source_name, p_idx)

                p_html = _assemble_page_html(p_page, p_img)
                p_spatial = _assemble_spatial_page_html(p_page, p_img)

                all_html_parts.append(
                    f'<div style="margin-bottom: 32px; padding-bottom: 20px; border-bottom: 2px dashed #003366;">'
                    f'<div style="font-size: 14px; font-weight: bold; color: #003366; margin-bottom: 12px; border-left: 4px solid #003366; padding-left: 8px;">📄 Page {p_idx} of {page_count}</div>'
                    f'{p_html}</div>'
                )
                all_spatial_parts.append(
                    f'<div style="margin-bottom: 32px;">'
                    f'<div style="font-size: 14px; font-weight: bold; color: #003366; margin-bottom: 12px; border-left: 4px solid #003366; padding-left: 8px;">📄 Spatial View — Page {p_idx} of {page_count}</div>'
                    f'{p_spatial}</div>'
                )

            prog_bar.progress(100, text=f"✅ All {page_count} pages extracted successfully!")
            full_html = "\n".join(all_html_parts)
            spatial_html = "\n".join(all_spatial_parts)

            docx_bytes = create_docx_from_surya_pages(
                pages=all_pages_data,
                document_title=doc_tags.get("document_title", "Tata Power Digitized Document"),
            )
            annotated = all_annotated[0][1]
            page = all_pages_data[0][0]
        else:
            # Single Page Processing Mode
            pdf_status = "Skipped (Image file)"
            if "pdf" in filetype:
                with st.spinner("Stage 1/4: Checking PDF text quality & vector structure..."):
                    pdf_status, _ = ocr_errors(in_file, page_count)
                st.info(f"📋 **Stage 1 (Pre-Flight Check)**: {pdf_status}")
            
            with st.spinner("Stage 2/4: Running Layout Analysis & Block Reading Order..."):
                annotated, page, layout, layout_time, block_time = block_ocr(pil_image)
            
            p_num = page_number or 1
            doc_tags = tag_document_page(page, pil_image, source_name, p_num)
            total_elapsed = layout_time + block_time

            full_html = _assemble_page_html(page, pil_image)
            spatial_html = _assemble_spatial_page_html(page, pil_image)
            docx_bytes = create_docx_from_surya_page(
                page=page,
                pil_image=pil_image,
                document_title=doc_tags.get("document_title", "Tata Power Digitized Document"),
            )

        tables_found = sum(1 for b in page.blocks if b.label in ("Table", "TableOfContents"))
        st.success(f"✅ Pipeline Completed in {total_elapsed:.2f}s!")
        
        tab_doc, tab_spatial, tab_tags, tab_export, tab_inspector = st.tabs([
            "📄 Presentable Document Preview",
            "🎯 1:1 Exact Spatial Layout Replica",
            "🏷️ Metadata Tags & Entities",
            "📥 Export Center (.docx / .html)",
            "🔍 Pipeline Diagnostic Inspector",
        ])
        
        with tab_doc:
            st.markdown(f"### ⚡ {doc_tags.get('document_title', 'Tata Power Digitized Document')}")
            st.caption(f"Document-agnostic flow ({doc_tags.get('document_type')}) with inline image graphics, tables, and section headings")
            render_ocr_html(full_html, height=700)
            
        with tab_spatial:
            st.markdown("### 🎯 1:1 Spatial Layout Replica")
            st.caption("Exact spatial coordinate positioning matching the original document page layout")
            render_ocr_html(spatial_html, height=750)

        with tab_tags:
            st.markdown("### 🏷️ Extracted Document Metadata & Classification Tags")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="📄 Document Title", value=doc_tags.get("document_title", "N/A"))
            with m2:
                st.metric(label="🏷️ Classified Document Type", value=doc_tags.get("document_type", "General Document"))
            
            st.markdown("#### 📝 Document Summary")
            st.info(doc_tags.get("summary", "No summary extracted."))
            
            entities = doc_tags.get("entities", [])
            if entities:
                st.markdown("#### 🔑 Key Extracted Entities")
                st.write(" • ".join([f"`{e}`" for e in entities]))
                
            st.markdown("#### 📊 Layout Block Distribution")
            st.json(doc_tags.get("layout_counts", {}))
            
        with tab_export:
            st.subheader("📥 Export Digitized Document")
            st.write("Download your extracted document in Microsoft Word (.docx) with embedded image graphics or HTML presentation format:")
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    label="📄 Download Word Document (.docx)",
                    data=docx_bytes,
                    file_name=f"{doc_tags.get('document_title', 'Tata_Power_Document').replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with col_exp2:
                st.download_button(
                    label="📝 Download HTML File (.html)",
                    data=full_html.encode("utf-8"),
                    file_name=f"{doc_tags.get('document_title', 'Tata_Power_Document').replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True,
                )
                
        with tab_inspector:
            st.image(
                annotated,
                caption="Pipeline Bounding Box & Reading Order Overlay",
                use_container_width=True,
            )
            with st.expander("Extracted Full Page HTML", expanded=False):
                st.code(full_html, language="html")
            for blk in page.blocks:
                with st.expander(f"#{blk.reading_order} {blk.label} (conf {blk.confidence:.2f})"):
                    if blk.skipped or blk.label in ("Picture", "Figure", "Image", "Diagram", "Logo", "Stamp", "Photo"):
                        cx0 = max(0, int(blk.bbox[0]) - 4)
                        cy0 = max(0, int(blk.bbox[1]) - 4)
                        cx1 = min(pil_image.size[0], int(blk.bbox[2]) + 4)
                        cy1 = min(pil_image.size[1], int(blk.bbox[3]) + 4)
                        if cx1 > cx0 and cy1 > cy0:
                            st.image(pil_image.crop((cx0, cy0, cx1, cy1)), caption=f"Extracted Image Region ({blk.label})")
                    else:
                        render_ocr_html(blk.html, height=160)


if run_text_det:
    det_img, text_pred, elapsed = text_detection(pil_image)
    with col1:
        _show_timing("Text detection", elapsed, f"{len(text_pred.bboxes)} polys")
        st.image(det_img, caption="Detected Text", use_container_width=True)
        st.json(
            text_pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=False
        )


if run_layout:
    annotated, pred, elapsed = layout_detection(pil_image, use_fast=use_fast_layout)
    with col1:
        label = "Layout (fast)" if use_fast_layout else "Layout"
        _show_timing(label, elapsed, f"{len(pred.bboxes)} blocks")
        st.image(annotated, caption="Detected Layout", use_container_width=True)
        st.json(pred.model_dump(), expanded=False)


if run_block_ocr:
    annotated, page, layout, t_layout, t_blocks = block_ocr(pil_image)
    with col1:
        n_blocks = len(page.blocks)
        n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
        _show_timing("Block OCR — layout", t_layout, f"{n_blocks} blocks")
        _show_timing("Block OCR — per-block OCR", t_blocks, f"{n_ok} OCR'd")
        _show_timing("Block OCR — total", t_layout + t_blocks)
        st.image(
            annotated,
            caption="Block OCR (green=ok, orange=skipped, red=error)",
            use_container_width=True,
        )
        full_html = _assemble_page_html(page)
        with st.expander("Full page HTML (rendered)", expanded=False):
            render_ocr_html(full_html, height=600)
        with st.expander("Full page HTML (source)", expanded=False):
            st.code(full_html, language="html")
        for blk in page.blocks:
            with st.expander(
                f"#{blk.reading_order} {blk.label} (conf {blk.confidence:.2f})"
            ):
                # Diagnostics: show numeric bbox + polygon + a thumbnail with the
                # drawn rectangle highlighted, then the actual crop fed to OCR.
                xs = [p[0] for p in blk.polygon]
                ys = [p[1] for p in blk.polygon]
                bbox_drawn = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                cx0 = max(0, int(min(xs)) - 4)
                cy0 = max(0, int(min(ys)) - 4)
                cx1 = min(pil_image.size[0], int(max(xs)) + 4)
                cy1 = min(pil_image.size[1], int(max(ys)) + 4)
                st.text(
                    f"bbox(drawn) = {bbox_drawn}\n"
                    f"crop(ocr)  = {(cx0, cy0, cx1, cy1)}  (= bbox ± 4px pad)"
                )
                # Thumbnail with this block's rectangle highlighted in red.
                thumb = pil_image.copy()
                ImageDraw.Draw(thumb).rectangle(bbox_drawn, outline="red", width=4)
                st.image(thumb, caption="this block's drawn rect (red)", width=300)
                # The actual crop fed to OCR
                if cx1 > cx0 and cy1 > cy0:
                    st.image(pil_image.crop((cx0, cy0, cx1, cy1)), caption="OCR crop")
                if blk.skipped:
                    st.info("Block skipped (visual label)")
                elif blk.error:
                    st.error("Block OCR errored")
                else:
                    render_ocr_html(blk.html, height=160)
                    st.code(blk.html, language="html")


if run_full_page_ocr:
    annotated, page, elapsed = full_page_ocr(pil_image)
    with col1:
        n_blocks = len(page.blocks)
        n_ok = sum(1 for b in page.blocks if not b.skipped and not b.error)
        _show_timing("Surya 2 Native VLM (MacBook)", elapsed, f"{n_blocks} blocks parsed, {n_ok} OK")
        
        full_html = _assemble_page_html(page, pil_image)
        spatial_html = _assemble_spatial_page_html(page, pil_image)
        
        # Generate DOCX binary using Surya native page blocks and cropped image regions
        docx_bytes = create_docx_from_surya_page(
            page=page,
            pil_image=pil_image,
            document_title="Tata Power Digitized Document",
        )
        
        tab_doc, tab_spatial, tab_export, tab_inspector = st.tabs([
            "📄 Presentable Document Preview",
            "🎯 1:1 Exact Spatial Layout Replica",
            "📥 Export Center (.docx / .html)",
            "🔍 Layout Overlay & Inspector",
        ])
        
        with tab_doc:
            st.markdown("### ⚡ Tata Power Digitized Document")
            st.caption("Extracted document flow layout with embedded image graphics, formatted tables, and section headings")
            render_ocr_html(full_html, height=700)
            
        with tab_spatial:
            st.markdown("### 🎯 1:1 Spatial Layout Replica")
            st.caption("Exact spatial coordinate positioning matching the original document page layout")
            render_ocr_html(spatial_html, height=750)
            
        with tab_export:
            st.subheader("📥 Export Digitized Document")
            st.write("Download your extracted document in Microsoft Word (.docx) with embedded image graphics or HTML presentation format:")
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button(
                    label="📄 Download Word Document (.docx)",
                    data=docx_bytes,
                    file_name="Tata_Power_Digitized_Document.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with col_exp2:
                st.download_button(
                    label="📝 Download HTML File (.html)",
                    data=full_html.encode("utf-8"),
                    file_name="Tata_Power_Digitized_Document.html",
                    mime="text/html",
                    use_container_width=True,
                )
                
        with tab_inspector:
            st.image(
                annotated,
                caption="Full-Page OCR Layout Overlay (green=ok, orange=skipped, red=error)",
                use_container_width=True,
            )
            with st.expander("Full page HTML (source code)", expanded=False):
                st.code(full_html, language="html")
            for blk in page.blocks:
                with st.expander(
                    f"#{blk.reading_order} {blk.label} (conf {blk.confidence:.2f})"
                ):
                    if blk.skipped or blk.label in ("Picture", "Figure", "Image", "Diagram", "Logo", "Stamp", "Photo"):
                        cx0 = max(0, int(blk.bbox[0]) - 4)
                        cy0 = max(0, int(blk.bbox[1]) - 4)
                        cx1 = min(pil_image.size[0], int(blk.bbox[2]) + 4)
                        cy1 = min(pil_image.size[1], int(blk.bbox[3]) + 4)
                        if cx1 > cx0 and cy1 > cy0:
                            st.image(pil_image.crop((cx0, cy0, cx1, cy1)), caption=f"Extracted Image Region ({blk.label})")
                        if blk.skipped:
                            st.info("Visual Block (Cropped & Embedded)")
                    elif blk.error:
                        st.error("Block OCR errored")
                    else:
                        render_ocr_html(blk.html, height=160)
                        st.code(blk.html, language="html")


if run_table_rec:
    table_img, preds, t_layout, t_table = table_recognition(
        pil_image, table_mode, skip_table_detection, use_fast_layout=use_fast_layout
    )
    with col1:
        if not skip_table_detection:
            _show_timing("Table Rec — layout", t_layout, f"{len(preds)} tables found")
        _show_timing(f"Table Rec — {table_mode}", t_table)
        if not skip_table_detection:
            _show_timing("Table Rec — total", t_layout + t_table)
        st.image(table_img, caption="Table Recognition", use_container_width=True)
        for pred in preds:
            if pred.mode == "full" and pred.html:
                with st.expander("Table HTML"):
                    render_ocr_html(pred.html, height=400)
                    st.code(pred.html, language="html")
            else:
                st.json(pred.model_dump(), expanded=False)


if run_ocr_errors:
    if "pdf" not in filetype:
        st.error("This feature only works with PDFs.")
    else:
        label, results = ocr_errors(in_file, page_count)
        with col1:
            st.write(label)
            st.json(results)


with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)


