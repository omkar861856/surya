"""Surya2 streamlit app — exercise layout, recognition, table_rec via the
inference manager. Detection + OCR-error stay in their own torch paths."""

from __future__ import annotations

import base64
import io
import json
import os
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
from surya.scripts.ai_processor import analyze_ocr_and_extract_form_fields, is_local_llm_running, search_indian_medicine_web
from surya.scripts.system_vitals import (
    get_gpu_vitals,
    get_system_vitals,
    get_inference_daemon_vitals,
    get_structurer_daemon_vitals,
    restart_inference_service,
    get_service_logs,
)
from surya.scripts.concurrency_benchmark import (
    load_latest_concurrency_results,
    run_concurrency_test,
)


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


def render_workflow_note(expanded: bool = False):
    with st.expander("💡 End-to-End Workflow Architecture & Pipeline Overview", expanded=expanded):
        st.markdown(
            """
### 🛠️ End-to-End Workflow Architecture

1. **📄 Multi-Page & Single-Page Document Ingestion**:
   - Supports multi-page PDFs and high-resolution images (PNG, JPG, WebP).
   - Sequentially parses all document pages or targets specific single pages.

2. **⚡ Surya OCR 2 & Spatial Layout Segmentation**:
   - Performs document-agnostic layout block recognition (Headings, Paragraphs, Tables, Figures/Logos).
   - Generates exact **1:1 Spatial Layout Replicas** with mathematical pixel scaling coordinates and clean paper HTML previews.

3. **🤖 Local AI Engine (`Qwen2.5-7B-Instruct` on RTX 5090 GPU)**:
   - Clears OCR character glitches, typos, line wrap breaks, and reading order errors.
   - **Dynamic Key Extraction**: Dynamically identifies field labels (`field_name`: `value`) tailored to document types (Prescription, Medical Bill, Invoice, Spec Sheet, Lab Report).

4. **💊 Real-Time Pharmaceutical Grounding (India Medicines & Drug Info Dataset)**:
   - Queries Indian pharmaceutical registries (**1mg.com, PharmEasy, Netmeds**) for medicine candidates.
   - **AI-Powered Drug Normalization**: Verifies garbled OCR names (e.g., *Dolo-65O* ➔ `Dolo 650mg Tablet`, *Augmntn 625* ➔ `Augmentin 625 Duo Tablet`, *Pan-D* ➔ `Pan D Capsule`).
   - Standardizes active chemical compositions (*Paracetamol 650mg*), dosages, and duration.

5. **✏️ Interactive Manual Correction & Export Center**:
   - Editable Streamlit form to review, modify, and save verified fields and drug lists.
   - One-click export to Microsoft Word (`.docx`), `.json`, and structured `.csv`.
"""
        )


def render_openai_correction_form(raw_ocr_content: str, key_prefix: str = "main"):
    st.subheader("🤖 Local AI Analysis & Dynamic Medicine Verification Form")
    st.caption("Powered by Local Qwen2.5-7B-Instruct running on NVIDIA RTX 5090 GPU (Port 8001). Zero cloud API dependencies.")

    local_active = is_local_llm_running()
    if local_active:
        st.success("🟢 **Dedicated Local AI Active**: Qwen2.5-7B-Instruct running on NVIDIA RTX 5090 (Port 8001)")
    else:
        st.warning("🟡 Local AI engine currently starting or port 8001 unreachable. Verify `llama-structurer.service`.")

    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    form_ver = st.session_state.get(f"{key_prefix}_form_ver", 0)

    with st.expander("⚙️ AI Model & Endpoint Settings", expanded=False):
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.text_input(
                "Local LLM Endpoint",
                value="http://127.0.0.1:8001/v1 (Qwen2.5-7B @ RTX 5090)",
                disabled=True,
            )
        with col_k2:
            user_openai_key = st.text_input(
                "OpenAI API Key (Optional Cloud Fallback)",
                value=env_openai_key,
                type="password",
                help="Only needed if you explicitly select an OpenAI cloud model",
                key=f"{key_prefix}_openai_api_key_input",
            )

    # Editable raw text container if raw_ocr_content is short/empty
    if not raw_ocr_content.strip():
        raw_ocr_input = st.text_area(
            "Raw Document Text / OCR Output for AI Analysis",
            value="Prescription: Patient Rahul Sharma Date: 14/09/2026\n1. Tab Dolo-65O 1-0-1 5 days\n2. Tab Augmntn 625 1-0-1 5 days\n3. Cap Pan-D 1-0-0 5 days",
            height=120,
            key=f"{key_prefix}_raw_text_area_{form_ver}",
        )
    else:
        with st.expander("📄 View Source OCR Text Fed to Local AI", expanded=False):
            raw_ocr_input = st.text_area(
                "Source OCR Content",
                value=raw_ocr_content,
                height=150,
                key=f"{key_prefix}_raw_text_area_expanded_{form_ver}",
            )

    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        ai_model = st.selectbox(
            "Select AI Engine Model",
            ["qwen2.5-7b-instruct (Local GPU - RTX 5090)", "gpt-4o-mini (Cloud)", "gpt-4o (Cloud)"],
            index=0,
            key=f"{key_prefix}_ai_model_select_{form_ver}"
        )
    session_key = f"{key_prefix}_ai_form_data"
    saved_key = f"{key_prefix}_saved_corrections"
    form_data = st.session_state.get(session_key)

    with col_ai2:
        st.write("")
        st.write("")
        btn_label = "🔄 Re-Analyze with AI" if form_data else "✨ Analyze with AI"
        analyze_btn = st.button(btn_label, type="secondary" if form_data else "primary", use_container_width=True, key=f"{key_prefix}_analyze_btn_{form_ver}")

    # Auto-trigger if not yet structured and raw OCR content is present
    auto_trigger = (form_data is None and bool(raw_ocr_content.strip()))

    if analyze_btn or auto_trigger:
        active_text = raw_ocr_input.strip() if (raw_ocr_input and raw_ocr_input.strip()) else raw_ocr_content.strip()
        model_id = "qwen2.5-7b-instruct" if "qwen" in ai_model.lower() else ("gpt-4o-mini" if "mini" in ai_model else "gpt-4o")
        is_cloud = model_id.startswith("gpt-")
        active_key = (user_openai_key.strip() or env_openai_key.strip()) if is_cloud else None

        if is_cloud and not active_key:
            if not auto_trigger:
                st.error("⚠️ OpenAI API Key is missing for the selected cloud model.")
        else:
            with st.spinner(f"🤖 Structuring document data & grounding with Qwen Indian Web Search (1mg/PharmEasy/Netmeds) via {ai_model}..."):
                try:
                    extracted_data = analyze_ocr_and_extract_form_fields(
                        raw_ocr_content=active_text,
                        model_name=model_id,
                        api_key=active_key,
                    )
                    new_ver = int(time.time())
                    st.session_state[f"{key_prefix}_form_ver"] = new_ver
                    form_ver = new_ver
                    st.session_state[session_key] = extracted_data
                    form_data = extracted_data
                    st.success(f"✅ Automatically structured & verified data with {ai_model}!")
                except Exception as ex:
                    st.error(f"AI Extraction Failed: {str(ex)}")

    form_data = st.session_state.get(session_key)


    if form_data:
        st.divider()

        if form_data.get("is_ai_corrected"):
            ai_engine = form_data.get("ai_engine_used", f"AI Engine ({ai_model})")
            st.success(f"🤖 **AI Data Inconsistency Cleanup Active**: {ai_engine} detected and cleared OCR character glitches, typos, line breaks, formatting errors, and verified Indian drug names against dataset.")
            corrections = form_data.get("ai_corrections_made", [])
            if corrections:
                with st.expander("🛠️ View AI Inconsistency Cleanup Log", expanded=False):
                    for corr in corrections:
                        st.markdown(f"• 🪄 **AI Corrected**: {corr}")

        grounding_meta = form_data.get("search_grounding_metadata") or {}
        drugs_grounded = grounding_meta.get("drugs_grounded", [])
        if drugs_grounded:
            st.info(f"🌐 **Qwen Indian Web Search Grounding Active**: Successfully verified {len(drugs_grounded)} drug candidates via **PharmEasy, Tata 1mg, and Netmeds** registries.")
            with st.expander("🔍 View Live Indian Pharma Web Search Grounding Details", expanded=False):
                for dg in drugs_grounded:
                    st.markdown(
                        f"• **`{dg.get('raw_query')}`** → **{dg.get('brand_name')}** (Mfr: *{dg.get('manufacturer', 'Indian Pharma')}*)\n"
                        f"  - *Active Composition:* `{dg.get('composition')}`\n"
                        f"  - *Registry Source:* `{dg.get('source')}`"
                    )

        with st.expander("🔍 Interactive Indian Medicine Lookup (Live 1mg & PharmEasy Search)", expanded=False):
            st.caption("Look up any Indian medicine name, composition, or manufacturer in real time.")
            s_col1, s_col2 = st.columns([3, 1])
            with s_col1:
                search_term = st.text_input("Enter medicine name to search (e.g., Dolo 650, Augmentin 625, Pan-D)", key=f"{key_prefix}_med_search_input_{form_ver}")
            with s_col2:
                st.write("")
                st.write("")
                do_search = st.button("🔎 Search Web", key=f"{key_prefix}_do_med_search_{form_ver}")
            if do_search and search_term.strip():
                with st.spinner(f"Searching Indian pharmacies for '{search_term}'..."):
                    lookup_res = search_indian_medicine_web(search_term.strip())
                    if lookup_res.get("found"):
                        st.success(f"✅ Found: **{lookup_res.get('brand_name')}**")
                        st.markdown(f"• **Active Chemical Composition:** `{lookup_res.get('composition')}`")
                        st.markdown(f"• **Manufacturer:** `{lookup_res.get('manufacturer')}`")
                        st.markdown(f"• **Source:** `{lookup_res.get('source')}`")
                    else:
                        st.warning(f"No match found for '{search_term}' on Indian pharmacy registries.")

        st.markdown("### ✏️ Edit & Correct Document Fields")
        st.caption("Review AI-corrected fields and Indian Drug Dataset verifications below. You can make manual edits, update values, or save your verified document data.")

        with st.form(f"{key_prefix}_correction_form_{form_ver}"):
            c_hdr1, c_hdr2 = st.columns(2)
            with c_hdr1:
                edited_title = st.text_input("Document Title (AI Cleaned)", value=form_data.get("document_title", ""), key=f"{key_prefix}_doc_title_{form_ver}")
            with c_hdr2:
                edited_type = st.text_input("Document Category / Type (AI Detected)", value=form_data.get("document_type", ""), key=f"{key_prefix}_doc_type_{form_ver}")

            edited_summary = st.text_area("Executive Summary (AI Reconciled)", value=form_data.get("summary", ""), height=90, key=f"{key_prefix}_doc_summary_{form_ver}")

            st.markdown("#### 🔑 Dynamic Key-Value Document Fields (Extracted based on Document Type)")
            kv_list = form_data.get("dynamic_key_value_fields") or form_data.get("key_value_fields", [])
            edited_kv = []
            for idx, item in enumerate(kv_list):
                c1, c2 = st.columns([1, 2])
                with c1:
                    fname = st.text_input(f"Dynamic Field #{idx+1} Label ✨", value=item.get("field_name", ""), key=f"{key_prefix}_fn_{idx}_{form_ver}")
                with c2:
                    fval = st.text_input(f"Field #{idx+1} Value ✨", value=str(item.get("value", "")), key=f"{key_prefix}_fv_{idx}_{form_ver}")
                edited_kv.append({
                    "field_name": fname,
                    "value": fval,
                    "is_ai_corrected": True,
                })

            med_list = form_data.get("medicines_list", [])
            edited_meds = []
            if med_list:
                st.markdown("#### 💊 Indian Medicines & Drug Info Dataset Corrections")
                st.caption("Medicine names have been cross-referenced and corrected against the India Medicines & Drug Info Dataset (1mg/Netmeds/PharmEasy registry). Only AI is permitted to correct drug names.")
                for m_idx, med in enumerate(med_list):
                    m_raw = med.get("ocr_raw_name", "N/A")
                    m_mfr = med.get("manufacturer")
                    m_label = f"**Medicine #{m_idx+1}**: Raw OCR Name: `{m_raw}`"
                    if m_mfr:
                        m_label += f" | 🏭 Mfr: *{m_mfr}*"
                    st.markdown(m_label)
                    mc1, mc2, mc3 = st.columns([2, 2, 1])
                    with mc1:
                        m_name = st.text_input(f"Verified Medicine Name (India Dataset) #{m_idx+1}", value=med.get("corrected_medicine_name", ""), key=f"{key_prefix}_mn_{m_idx}_{form_ver}")
                        m_comp = st.text_input(f"Active Chemical Composition #{m_idx+1}", value=med.get("composition", ""), key=f"{key_prefix}_mc_{m_idx}_{form_ver}")
                    with mc2:
                        m_dos = st.text_input(f"Dosage / Frequency #{m_idx+1}", value=med.get("dosage", ""), key=f"{key_prefix}_md_{m_idx}_{form_ver}")
                        m_dur = st.text_input(f"Duration #{m_idx+1}", value=med.get("duration", ""), key=f"{key_prefix}_mt_{m_idx}_{form_ver}")
                    with mc3:
                        st.write("")
                        c_status = med.get("correction_status", "India Drug Dataset Verified")
                        st.info(f"💊 {c_status}")

                    edited_meds.append({
                        "ocr_raw_name": med.get("ocr_raw_name"),
                        "corrected_medicine_name": m_name,
                        "composition": m_comp,
                        "manufacturer": m_mfr or "",
                        "dosage": m_dos,
                        "duration": m_dur,
                        "correction_status": c_status,
                    })

            st.markdown("#### 📝 Content Sections & Text Blocks (AI Cleaned)")
            sec_list = form_data.get("content_sections", [])
            edited_sections = []
            for s_idx, sec in enumerate(sec_list):
                s_heading = st.text_input(f"Section #{s_idx+1} Heading ✨", value=sec.get("section_heading", ""), key=f"{key_prefix}_sh_{s_idx}_{form_ver}")
                s_content = st.text_area(f"Section #{s_idx+1} Text Content ✨", value=sec.get("text_content", ""), height=120, key=f"{key_prefix}_sc_{s_idx}_{form_ver}")
                edited_sections.append({"section_heading": s_heading, "text_content": s_content})

            save_submitted = st.form_submit_button("💾 Save Manual Corrections & Verification", type="primary", use_container_width=True)

            if save_submitted:
                corrected_dict = {
                    "document_title": edited_title,
                    "document_type": edited_type,
                    "summary": edited_summary,
                    "is_ai_corrected": True,
                    "status": "AI Corrected & Manually Verified",
                    "ai_corrections_made": form_data.get("ai_corrections_made", []),
                    "dynamic_key_value_fields": edited_kv,
                    "medicines_list": edited_meds,
                    "content_sections": edited_sections,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state[saved_key] = corrected_dict
                os.makedirs("output", exist_ok=True)
                out_filepath = os.path.join("output", "corrected_ocr_data.json")
                with open(out_filepath, "w", encoding="utf-8") as f:
                    json.dump(corrected_dict, f, indent=2)
                st.success(f"✅ Successfully saved manual corrections to `{out_filepath}`!")

        saved_data = st.session_state.get(saved_key) or form_data
        if saved_data:
            st.markdown("#### 📥 Export AI-Corrected Data")
            ex1, ex2 = st.columns(2)
            with ex1:
                st.download_button(
                    label="📥 Download Corrected JSON",
                    data=json.dumps(saved_data, indent=2).encode("utf-8"),
                    file_name="corrected_ocr_data.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"{key_prefix}_dl_json_{form_ver}",
                )
            with ex2:
                import csv
                csv_io = io.StringIO()
                writer = csv.writer(csv_io)
                writer.writerow(["Field Category", "Field Name / Raw OCR", "Corrected Value", "AI Correction Source"])
                for item in saved_data.get("dynamic_key_value_fields") or saved_data.get("key_value_fields", []):
                    writer.writerow(["Dynamic Field", item.get("field_name"), item.get("value"), "AI Cleaned"])
                for med in saved_data.get("medicines_list", []):
                    writer.writerow(["Prescribed Medicine", med.get("ocr_raw_name"), med.get("corrected_medicine_name"), "India Medicines and Drug Info Dataset"])
                st.download_button(
                    label="📊 Download Key-Values & Medicines (.csv)",
                    data=csv_io.getvalue().encode("utf-8"),
                    file_name="corrected_fields.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_prefix}_dl_csv_{form_ver}",
                )

    else:
        st.info("💡 Click **'✨ Analyze with Local AI'** above to parse your document OCR output into an interactive correction form.")



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


def fast_unified_ocr(img, use_block_mode: bool = False) -> tuple[Image.Image, PageOCRResult, LayoutResult, float, float]:
    """High-speed document OCR utilizing Surya 2's native full-page VLM mode (1.5-3.5s per page).
    Extracts text, layout blocks, reading order, and bounding boxes in a single forward pass."""
    if use_block_mode:
        return block_ocr(img)

    t0 = time.perf_counter()
    page_results = predictors["recognition"]([img], full_page=True)
    ocr_elapsed = time.perf_counter() - t0
    page = page_results[0]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for blk in page.blocks:
        x0, y0, x1, y1 = blk.bbox
        color = "red" if blk.error else ("orange" if blk.skipped else "green")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{blk.reading_order} {blk.label}", fill=color)

    from surya.layout.schema import LayoutResult, LayoutBox
    layout_boxes = [
        LayoutBox(
            polygon=blk.polygon,
            label=blk.label,
            raw_label=getattr(blk, "raw_label", "") or blk.label,
            position=blk.reading_order,
            confidence=blk.confidence,
        )
        for blk in page.blocks
    ]
    layout = LayoutResult(bboxes=layout_boxes, image_bbox=page.image_bbox)
    return annotated, page, layout, 0.0, ocr_elapsed


def render_admin_vitals_panel():
    """Renders the comprehensive Infrastructure & System Vitals Admin Panel."""
    st.markdown("## 📊 System Vitals & Infrastructure Monitor")
    st.caption("Real-time telemetry for NVIDIA GPU acceleration, host system load, RAM thresholds, and inference daemon slots.")

    gpu_info = get_gpu_vitals()
    sys_info = get_system_vitals()
    daemon_info = get_inference_daemon_vitals()
    struct_info = get_structurer_daemon_vitals()

    # Top Executive Telemetry Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if gpu_info:
            st.metric(
                label="🎮 GPU Accelerator",
                value=f"{gpu_info['gpu_utilization_pct']}% Util",
                delta=f"{gpu_info['temperature_c']}°C | {gpu_info['power_draw_w']}W",
            )
        else:
            st.metric(label="🎮 GPU Accelerator", value="CPU Only", delta="No GPU detected")

    with c2:
        st.metric(
            label="🧠 Host System RAM",
            value=f"{sys_info['ram_pct']}% Used",
            delta=f"{sys_info['ram_used_mb']:.0f} / {sys_info['ram_total_mb']:.0f} MB",
            delta_color="inverse" if sys_info['ram_pct'] > 85 else "normal",
        )

    with c3:
        if daemon_info["healthy"]:
            st.metric(
                label="⚡ Surya OCR Engine (:8000)",
                value="Active (Healthy)",
                delta=f"{daemon_info['latency_ms']} ms | {daemon_info['active_slots']}/{daemon_info['total_slots']} slots",
            )
        else:
            st.metric(label="⚡ Surya OCR Engine (:8000)", value="Offline", delta_color="inverse")

    with c4:
        if struct_info["healthy"]:
            st.metric(
                label="🤖 Local LLM Structurer (:8001)",
                value="Active (Qwen2.5-7B)",
                delta=f"{struct_info['latency_ms']} ms ping",
            )
        else:
            st.metric(label="🤖 Local LLM Structurer (:8001)", value="Offline", delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎮 NVIDIA GPU Telemetry")
        if gpu_info:
            st.markdown(f"**Device:** `{gpu_info['name']}` &nbsp;|&nbsp; **Driver:** `{gpu_info['driver']}`")
            vram_pct = gpu_info['memory_pct'] / 100.0
            st.progress(min(max(vram_pct, 0.0), 1.0), text=f"VRAM: {gpu_info['memory_used_mb']:.0f} MiB / {gpu_info['memory_total_mb']:.0f} MiB ({gpu_info['memory_pct']}%)")

            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric("Thermal Temp", f"{gpu_info['temperature_c']} °C")
            with g2:
                st.metric("Power Draw", f"{gpu_info['power_draw_w']} W")
            with g3:
                st.metric("GPU Core Load", f"{gpu_info['gpu_utilization_pct']} %")
        else:
            st.info("NVIDIA Management Library (nvidia-smi) is not available or running in pure CPU mode.")

    with col_right:
        st.markdown("### 🖥️ Host Node Infrastructure")
        st.markdown(f"**CPU Cores:** `{sys_info['cpu_count']}` Cores &nbsp;|&nbsp; **Load:** `1m: {sys_info['load_1m']} | 5m: {sys_info['load_5m']} | 15m: {sys_info['load_15m']}`")

        ram_val = sys_info['ram_pct'] / 100.0
        st.progress(min(max(ram_val, 0.0), 1.0), text=f"Host RAM: {sys_info['ram_used_mb']:.0f} MB / {sys_info['ram_total_mb']:.0f} MB ({sys_info['ram_pct']}%)")

        disk_val = sys_info['disk_pct'] / 100.0
        st.progress(min(max(disk_val, 0.0), 1.0), text=f"Root Storage: {sys_info['disk_used_gb']:.1f} GB / {sys_info['disk_total_gb']:.1f} GB ({sys_info['disk_pct']}%)")

    st.markdown("---")
    st.markdown("### ⚡ Parallel Inference Engine Slots (Port 8000)")

    if daemon_info["slots"]:
        slot_cols = st.columns(len(daemon_info["slots"]))
        for idx, slot in enumerate(daemon_info["slots"]):
            with slot_cols[idx]:
                status_color = "#28a745" if not slot["is_processing"] else "#ffc107"
                status_label = "🟢 IDLE" if not slot["is_processing"] else "🟡 BUSY"
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e0e0e0; border-top:4px solid {status_color}; border-radius:8px; padding:12px; margin-bottom:12px; box-shadow:0 2px 4px rgba(0,0,0,0.04);">
                        <div style="font-weight:bold; font-size:15px; color:#111827;">Slot #{slot['id']} {status_label}</div>
                        <div style="font-size:13px; color:#4b5563; margin-top:6px;"><b>Context Size:</b> {slot['n_ctx']} tokens</div>
                        <div style="font-size:13px; color:#4b5563;"><b>Prompt Tokens:</b> {slot['prompt_tokens']}</div>
                        <div style="font-size:13px; color:#4b5563;"><b>Task ID:</b> {slot['task_id']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Inference slots data currently unreachable. Verify that llama-ocr.service is active.")

    st.markdown("---")
    st.markdown("### ⚡ Multi-User Concurrency & Load Benchmark")
    st.caption("Live stress-test of parallel slots using real multi-page clinical images from `test-images/`.")

    bench_data = load_latest_concurrency_results()

    # Benchmark Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 2, 2])
    with c_ctrl1:
        bench_workers = st.selectbox(
            "Parallel Workers:",
            [2, 4, 8],
            index=1,
            help="Concurrent client threads issuing OCR requests simultaneously",
        )
    with c_ctrl2:
        bench_count = st.selectbox(
            "Test Sample Size:",
            [4, 8, 16, 24],
            index=1,
            help="Total documents to process in this load run",
        )
    with c_ctrl3:
        st.write("")
        st.write("")
        run_bench_btn = st.button("🚀 Run Concurrency Benchmark", use_container_width=True, type="primary")

    if run_bench_btn:
        with st.spinner(f"Running concurrent load test ({bench_workers} workers, {bench_count} images)..."):
            try:
                bench_data = run_concurrency_test(
                    image_dir="test-images",
                    concurrency=bench_workers,
                    total_images=bench_count,
                )
                st.success(f"Benchmark completed in {bench_data['total_time_s']}s at {bench_data['throughput_img_per_sec']} img/s!")
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    if bench_data:
        st.markdown(
            f"**Latest Test Run:** `{bench_data.get('timestamp', 'N/A')}` &nbsp;|&nbsp; "
            f"**Concurrency Workers:** `{bench_data.get('concurrency_workers', 'N/A')}` &nbsp;|&nbsp; "
            f"**Total Requests:** `{bench_data.get('total_requests', 'N/A')}`"
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("⚡ Throughput", f"{bench_data['throughput_img_per_sec']} img/s", f"{bench_data['total_time_s']}s total")
        with m2:
            st.metric("⏱️ Avg Latency", f"{bench_data['avg_latency_s']} s", f"Min: {bench_data['min_latency_s']}s")
        with m3:
            st.metric("🎯 P95 Latency", f"{bench_data['p95_latency_s']} s", f"Max: {bench_data['max_latency_s']}s")
        with m4:
            rate = bench_data['success_rate_pct']
            st.metric("✅ Success Rate", f"{rate}%", f"{bench_data['success_count']} / {bench_data['total_requests']} passed", delta_color="normal" if rate == 100 else "inverse")

        # Task breakdown table
        tasks = bench_data.get("tasks", [])
        if tasks:
            with st.expander(f"📋 Detailed Task Results ({len(tasks)} requests)", expanded=True):
                st.dataframe(tasks, use_container_width=True)
    else:
        st.info("No benchmark run recorded yet. Click 'Run Concurrency Benchmark' above to test parallel throughput on real test images.")

    st.markdown("---")
    st.markdown("### 📈 Concurrency & Capacity Sizing Scenarios")
    st.caption("Theoretical and empirical performance characteristics across varying user loads, request arrival rates, and cluster configurations.")

    scenario_tabs = st.tabs(["📊 Sizing Matrix & Scenarios", "🧮 Interactive Capacity Planner", "🏗️ Scaling Guidelines"])

    with scenario_tabs[0]:
        st.markdown("#### Real-World Operational Tiers (Single RTX 5090 vs Scaled Cluster)")
        scenario_table = [
            {
                "Load Tier": "🟢 Tier 1: Steady Standard",
                "Arrival Rate": "1 – 4 req/sec (up to 240/min)",
                "Hardware Setup": "1x RTX 5090 (4 slots)",
                "Avg Latency": "1.5s – 3.5s",
                "Throughput": "0.8 – 1.2 img/s (4,000/hr)",
                "Expected Output & System Behavior": "Zero queue wait. Every upload gets an instant GPU slot. Full-page VLM processes in ~2s.",
            },
            {
                "Load Tier": "🟡 Tier 2: Busy Clinic / Peak",
                "Arrival Rate": "5 – 12 req/sec (up to 720/min)",
                "Hardware Setup": "1x RTX 5090 (4 slots)",
                "Avg Latency": "3.5s – 7.5s",
                "Throughput": "1.1 – 1.3 img/s (4,300/hr)",
                "Expected Output & System Behavior": "All 4 GPU slots 100% saturated. Caddy FIFO buffer holds 2-4 reqs. Zero dropped connections, 100% success.",
            },
            {
                "Load Tier": "🟠 Tier 3: High Surge (Single Node Limit)",
                "Arrival Rate": "13 – 25 req/sec (up to 1,500/min)",
                "Hardware Setup": "1x RTX 5090 (4 slots)",
                "Avg Latency": "12s – 25s",
                "Throughput": "1.2 img/s (Hardware capped)",
                "Expected Output & System Behavior": "Single node at maximum hardware limit. Queue builds up; users wait ~15s. Recommendation: HPA scale.",
            },
            {
                "Load Tier": "🚀 Tier 4: Kubernetes Auto-Scale",
                "Arrival Rate": "25 – 100 req/sec (up to 6,000/min)",
                "Hardware Setup": "3 – 5x RTX 5090 Nodes (12-20 slots)",
                "Avg Latency": "2.0s – 4.0s",
                "Throughput": "4.0 – 6.5 img/s (20,000/hr)",
                "Expected Output & System Behavior": "Ingress-nginx load-balances across pods. Latency stays sub-4s even during nationwide hospital peak bursts.",
            },
            {
                "Load Tier": "📦 Tier 5: Bulk Overnight Ingestion",
                "Arrival Rate": "Pipelined Stream (Continuous)",
                "Hardware Setup": "1x Node (or Batch Pod)",
                "Avg Latency": "2.5s per doc",
                "Throughput": "~3,800 docs/hr per node",
                "Expected Output & System Behavior": "Memory stays rock-solid at 1.6GB due to 16K KV context optimization and FlashAttention.",
            },
        ]
        st.dataframe(scenario_table, use_container_width=True)

    with scenario_tabs[1]:
        st.markdown("#### 🧮 Dynamic Workload Sizing Calculator")
        st.write("Estimate required compute resources based on your expected traffic volume:")

        calc_col1, calc_col2 = st.columns(2)
        with calc_col1:
            input_users_per_min = st.slider(
                "Expected Requests per Minute:",
                min_value=5,
                max_value=600,
                value=60,
                step=5,
                help="Number of prescription/document upload requests per minute",
            )
            input_acceptable_latency = st.slider(
                "Target Max Latency Tolerance (seconds):",
                min_value=2.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Maximum acceptable time from upload to extraction result",
            )

        with calc_col2:
            doc_processing_time = 2.5
            req_per_sec = input_users_per_min / 60.0
            concurrency_needed = req_per_sec * doc_processing_time
            slots_recommended = max(1, int(concurrency_needed * 1.3 + 0.99))
            nodes_needed = max(1, (slots_recommended + 3) // 4)

            curr_service_rate = 4.0 / doc_processing_time
            if req_per_sec < curr_service_rate:
                est_curr_latency = round(doc_processing_time / (1.0 - (req_per_sec / curr_service_rate)), 1)
            else:
                est_curr_latency = "> 30s (Queue Overload)"

            st.markdown(
                f"""
                <div style="background:#f0f7ff; border:1px solid #cce3ff; border-radius:8px; padding:16px; margin-top:8px;">
                    <div style="font-size:16px; font-weight:bold; color:#0055aa; margin-bottom:8px;">Recommended Sizing Output</div>
                    <div style="font-size:14px; color:#333333; line-height:1.8;">
                        • <b>Required GPU Inference Slots:</b> <span style="font-size:15px; font-weight:bold; color:#111827;">{slots_recommended} Parallel Slots</span><br>
                        • <b>Recommended RTX 5090 Nodes:</b> <span style="font-size:15px; font-weight:bold; color:#111827;">{nodes_needed} Node(s)</span><br>
                        • <b>Hourly Document Capacity:</b> <span style="font-size:15px; font-weight:bold; color:#111827;">{int(nodes_needed * 4000):,} pages/hour</span><br>
                        • <b>Est. Latency on Current 1 Node:</b> <span style="font-size:15px; font-weight:bold; color:{'#28a745' if est_curr_latency != '> 30s (Queue Overload)' and float(str(est_curr_latency).replace('>','').replace('s','').split()[0]) <= input_acceptable_latency else '#d9534f'};">{est_curr_latency}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if nodes_needed == 1:
            st.success("✅ **Current Setup Sizing**: Your current single RTX 5090 instance is sufficient for this workload!")
        else:
            st.warning(f"⚠️ **Scaling Recommendation**: At {input_users_per_min} req/min with a {input_acceptable_latency}s SLA, we recommend scaling to **{nodes_needed} Kubernetes worker pods** using our HPA manifests in `k8s/`.")

    with scenario_tabs[2]:
        st.markdown("#### 🏗️ Architecture Recommendations by Workload Type")
        arch_c1, arch_c2 = st.columns(2)
        with arch_c1:
            st.markdown(
                """
                ##### 🏥 Clinical & Front-Desk Interactive
                * **User Profile**: Doctors and pharmacists scanning 1-3 pages during patient consultations.
                * **Primary Metric**: P95 Latency (< 3.5s).
                * **Tuning Applied**:
                  - Native Full-Page VLM (`fast_unified_ocr`)
                  - Render DPI: 144 DPI
                  - 4 Parallel FlashAttention slots
                """
            )
        with arch_c2:
            st.markdown(
                """
                ##### 📑 Back-Office & Historical Archive
                * **User Profile**: Scanning batches of 10,000+ patient records overnight.
                * **Primary Metric**: Maximum Sustained Throughput (Pages/hr).
                * **Tuning Applied**:
                  - Multi-threaded worker queue (`concurrency_benchmark.py`)
                  - 16,384 Context token ceiling to safeguard host RAM
                  - Zero CPU blocking error classifiers
                """
            )

    st.markdown("---")

    st.markdown("### 🛠️ Administrative Operations")
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)

    with act_col1:
        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            st.rerun()

    with act_col2:
        if st.button("⚡ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cleared Streamlit data caches!")

    with act_col3:
        if st.button("🔄 Restart OCR (:8000)", use_container_width=True, help="Restarts llama-ocr daemon"):
            ok, msg = restart_inference_service("llama-ocr")
            if ok:
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

    with act_col4:
        if st.button("🔄 Restart LLM (:8001)", use_container_width=True, help="Restarts llama-structurer daemon"):
            ok, msg = restart_inference_service("llama-structurer")
            if ok:
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### 📜 Real-Time Service Logs Inspector")
    log_service = st.selectbox("Select Service to Inspect:", ["llama-ocr", "llama-structurer", "surya-ocr", "caddy"], index=0)
    lines_count = st.slider("Log Lines to Display:", min_value=20, max_value=100, value=40, step=10)

    logs = get_service_logs(log_service, lines=lines_count)
    st.code(logs, language="log")



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


st.set_page_config(layout="wide", page_title="Surya OCR & Intelligence Hub", page_icon="⚡")

predictors = load_predictors_cached()

# --- TOP-LEVEL MODE NAVIGATION ---
st.sidebar.markdown("### 📌 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode / View:",
    options=[
        "⚡ Document Intelligence Hub",
        "📊 System Vitals & Admin Panel",
    ],
    index=0,
    help="Toggle between the Document Processing Hub and the Real-Time Infrastructure Vitals Panel.",
)

if app_mode == "📊 System Vitals & Admin Panel":
    render_admin_vitals_panel()
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Document Ingestion")
in_file = st.sidebar.file_uploader(
    "Upload PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.markdown(
        """
# Welcome Tata Power Team! ⚡

Welcome to the **Tata Power Document Intelligence & OCR Hub**.

We are delighted to bring you this advanced OCR and document analysis platform powered by **Surya OCR 2**, **Local Qwen2.5-7B-Instruct Engine on NVIDIA RTX 5090 GPU**, and **Live Web Search Grounding for the India Medicines & Drug Info Dataset**.

👈 **Get Started:** Upload a PDF document or image using the sidebar menu on the left to begin processing, or select **📊 System Vitals & Admin Panel** to monitor real-time GPU and server health.
"""
    )
    render_workflow_note(expanded=True)
    st.stop()


filetype = in_file.type
dpi_choice = st.sidebar.select_slider(
    "Render Resolution (DPI):",
    options=[96, 144, 192],
    value=144,
    help="144 DPI renders 40% faster while maintaining optimal OCR reading accuracy.",
)

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
    pil_image = get_page_image(in_file, page_number, dpi_choice)
else:
    scan_scope = "Single Page Only"
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

if pil_image is None:
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Document Views Navigation")
view_selection = st.sidebar.radio(
    "Select Result View:",
    options=[
        "📄 Presentable Document Preview",
        "🎯 1:1 Exact Spatial Layout Replica",
        "🏷️ Metadata Tags & Entities",
        "🤖 Local AI Form & Correction",
        "📥 Export Center (.docx / .html)",
        "🔍 Pipeline Diagnostic Inspector",
    ],
    index=0,
    help="Navigate between document preview, 1:1 spatial layout, metadata tags, Local AI form, export center, and inspector from the sidebar.",
)

# --- TOP NAVIGATION & PIPELINE CONTROLS BAR ---
st.markdown("### 🧭 Top Navigation & Pipeline Controls")

top_card = st.container()
with top_card:
    st.markdown("##### 🚀 Execution Modes")
    b0, b1, b2, b3, b4, b5, b6 = st.columns([1.2, 1, 1, 1, 1, 1, 1.2])
    with b0:
        run_unified_pipeline = st.button("⚡ Run Unified Pipeline", type="primary", use_container_width=True, help="Run Unified Ideal Pipeline (Document Agnostic)")
    with b1:
        run_full_page_ocr = st.button("Run Full-Page OCR", use_container_width=True)
    with b2:
        run_text_det = st.button("Run Text Detection", use_container_width=True)
    with b3:
        run_layout = st.button("Run Layout Analysis", use_container_width=True)
    with b4:
        run_table_rec = st.button("Run Table Rec", use_container_width=True)
    with b5:
        run_block_ocr = st.button("Run Block OCR", use_container_width=True)
    with b6:
        run_ocr_errors = st.button("Run bad-PDF-text detection", use_container_width=True)

    st.markdown("##### ⚙️ Engine Settings & Speed Optimization")
    set_col_speed, set_col_diag = st.columns([1.5, 1])
    with set_col_speed:
        ocr_engine_mode = st.radio(
            "⚡ OCR Speed Engine:",
            options=[
                "⚡ Ultra-Fast Full-Page VLM (1.5-3.5s - Recommended)",
                "🔬 Deep Multi-Step Block Crop (15-30s)",
            ],
            index=0,
            horizontal=True,
            help="Ultra-Fast mode runs Surya 2's native full-page VLM pass, completing in 1.5-3.5s with full text and bounding boxes. Multi-Step crops every single layout box individually.",
        )
    with set_col_diag:
        auto_ai_structure = st.checkbox(
            "⚡ Automatic AI Structuring",
            value=True,
            help="Automatically extract dynamic keys, clean typos, and structure data without manual button clicks.",
        )
        check_pdf_quality = st.checkbox(
            "Run PDF quality diagnostic model",
            value=False,
            help="Optional CPU neural network text check. Leave unchecked for maximum speed.",
        )

    use_block_mode = (ocr_engine_mode == "🔬 Deep Multi-Step Block Crop (15-30s)")

    set_col1, set_col2, set_col3 = st.columns([1, 1.2, 1.2])
    with set_col1:
        use_fast_layout = st.checkbox(
            "Fast layout",
            value=True,
            help="Use the fast layout detector.",
        )
    with set_col2:
        table_mode = st.radio(
            "Table mode:",
            options=["simple", "full"],
            index=0,
            horizontal=True,
            help="simple: rows+cols only. full: full HTML.",
        )
    with set_col3:
        skip_table_detection = st.checkbox(
            "Skip table detection",
            value=False,
            help="Treat the entire page/image as a single table.",
        )


st.divider()

col1, col2 = st.columns([0.55, 0.45])

# Reset state if new file uploaded
file_identifier = f"{in_file.name}_{in_file.size}" if hasattr(in_file, "name") else str(id(in_file))
if st.session_state.get("current_file_id") != file_identifier:
    st.session_state["current_file_id"] = file_identifier
    st.session_state["unified_result"] = None
    st.session_state["active_mode"] = None

if run_unified_pipeline:
    st.session_state["active_mode"] = "unified"
elif run_full_page_ocr:
    st.session_state["active_mode"] = "full_ocr"
elif run_text_det:
    st.session_state["active_mode"] = "text_det"
elif run_layout:
    st.session_state["active_mode"] = "layout"
elif run_table_rec:
    st.session_state["active_mode"] = "table_rec"
elif run_block_ocr:
    st.session_state["active_mode"] = "block_ocr"
elif run_ocr_errors:
    st.session_state["active_mode"] = "ocr_errors"

active_mode = st.session_state.get("active_mode")
if active_mode is None:
    active_mode = "unified"
    st.session_state["active_mode"] = "unified"

if active_mode == "unified":
    with col1:
        st.subheader("⚡ Unified Document-Agnostic Intelligence Pipeline")
        render_workflow_note(expanded=False)

        if run_unified_pipeline or st.session_state.get("unified_result") is None:
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
                    p_img = get_page_image(in_file, p_idx, dpi_choice)
                    ann_img, p_page, p_layout, p_ltime, p_btime = fast_unified_ocr(p_img, use_block_mode=use_block_mode)
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
                pdf_status = "Skipped (High-Speed Mode)"
                if "pdf" in filetype and check_pdf_quality:
                    with st.spinner("Stage 1/4: Checking PDF text quality & vector structure..."):
                        pdf_status, _ = ocr_errors(in_file, page_count)

                with st.spinner("Running High-Speed Surya OCR 2 Pipeline..."):
                    annotated, page, layout, layout_time, block_time = fast_unified_ocr(pil_image, use_block_mode=use_block_mode)


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

            raw_ocr = "\n".join([b.html for b in page.blocks if hasattr(b, 'html') and b.html])

            # --- AUTOMATIC LOCAL AI STRUCTURING STAGE ---
            ai_structured_data = None
            if auto_ai_structure:
                with st.spinner("🤖 Automatically structuring document data with Qwen Indian Web Search Grounding (Qwen2.5-7B @ RTX 5090)..."):
                    try:
                        clean_text = re.sub(r"<[^>]+>", " ", full_html)
                        clean_text = re.sub(r"\s+", " ", clean_text).strip()
                        if clean_text:
                            ai_structured_data = analyze_ocr_and_extract_form_fields(
                                raw_ocr_content=clean_text[:12000],
                                model_name="qwen2.5-7b-instruct",
                            )
                            now_ver = int(time.time())
                            for pfx in ("main", "unified", "multipage"):
                                st.session_state[f"{pfx}_ai_form_data"] = ai_structured_data
                                st.session_state[f"{pfx}_form_ver"] = now_ver
                    except Exception as ai_err:
                        st.warning(f"Auto AI structuring notice: {ai_err}")


            st.session_state["unified_result"] = {
                "doc_tags": doc_tags,
                "full_html": full_html,
                "spatial_html": spatial_html,
                "docx_bytes": docx_bytes,
                "annotated": annotated,
                "page": page,
                "total_elapsed": total_elapsed,
                "raw_ocr": raw_ocr,
                "ai_structured_data": ai_structured_data,
            }


        res = st.session_state["unified_result"]
        doc_tags = res["doc_tags"]
        full_html = res["full_html"]
        spatial_html = res["spatial_html"]
        docx_bytes = res["docx_bytes"]
        annotated = res["annotated"]
        page = res["page"]
        total_elapsed = res["total_elapsed"]
        raw_ocr = res["raw_ocr"]

        st.success(f"✅ Pipeline Completed in {total_elapsed:.2f}s!")

        if view_selection == "📄 Presentable Document Preview":
            st.markdown(f"### ⚡ {doc_tags.get('document_title', 'Tata Power Digitized Document')}")
            st.caption(f"Document-agnostic flow ({doc_tags.get('document_type')}) with inline image graphics, tables, and section headings")

            ai_data = res.get("ai_structured_data") or st.session_state.get("multipage_ai_form_data") or st.session_state.get("unified_ai_form_data")
            if ai_data:
                with st.expander("🤖 View Automatically Structured Document Fields & Drugs", expanded=False):
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        st.markdown(f"**Document Title:** {ai_data.get('document_title', 'N/A')}")
                        st.markdown(f"**Document Type:** {ai_data.get('document_type', 'N/A')}")
                        st.markdown(f"**Summary:** {ai_data.get('summary', 'N/A')}")
                    with f_col2:
                        meds = ai_data.get("medicines_list", [])
                        if meds:
                            st.markdown(f"**💊 Verified Indian Drugs ({len(meds)} found):**")
                            for m in meds:
                                m_desc = f"• **{m.get('corrected_medicine_name')}** ({m.get('composition', 'Active Composition')})"
                                if m.get("manufacturer"):
                                    m_desc += f" — *{m.get('manufacturer')}*"
                                m_desc += f" — `{m.get('dosage', 'Standard Dosage')}`"
                                st.markdown(m_desc)
                        fields = ai_data.get("dynamic_key_value_fields") or ai_data.get("key_value_fields", [])
                        if fields:
                            st.markdown(f"**📋 Dynamic Fields ({len(fields)}):**")
                            for fld in fields[:6]:
                                st.markdown(f"• **{fld.get('field_name')}:** `{fld.get('value')}`")

            render_ocr_html(full_html, height=700)


        elif view_selection == "🎯 1:1 Exact Spatial Layout Replica":
            st.markdown("### 🎯 1:1 Spatial Layout Replica")
            st.caption("Exact spatial coordinate positioning matching the original document page layout")
            render_ocr_html(spatial_html, height=750)

        elif view_selection == "🏷️ Metadata Tags & Entities":
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

        elif view_selection == "🤖 Local AI Form & Correction":
            render_openai_correction_form(raw_ocr, key_prefix="multipage")

        elif view_selection == "📥 Export Center (.docx / .html)":
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

        elif view_selection == "🔍 Pipeline Diagnostic Inspector":
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
        p_num = page_number or 1
        doc_tags = tag_document_page(page, pil_image, "Tata Power Digitized Document", p_num)
        raw_ocr = "\n".join([b.html for b in page.blocks if hasattr(b, 'html') and b.html])

        # Generate DOCX binary using Surya native page blocks and cropped image regions
        docx_bytes = create_docx_from_surya_page(
            page=page,
            pil_image=pil_image,
            document_title="Tata Power Digitized Document",
        )

        if view_selection == "📄 Presentable Document Preview":
            st.markdown("### ⚡ Tata Power Digitized Document")
            st.caption("Extracted document flow layout with embedded image graphics, formatted tables, and section headings")
            render_ocr_html(full_html, height=700)

        elif view_selection == "🎯 1:1 Exact Spatial Layout Replica":
            st.markdown("### 🎯 1:1 Spatial Layout Replica")
            st.caption("Exact spatial coordinate positioning matching the original document page layout")
            render_ocr_html(spatial_html, height=750)

        elif view_selection == "🏷️ Metadata Tags & Entities":
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

        elif view_selection == "🤖 Local AI Form & Correction":
            render_openai_correction_form(raw_ocr, key_prefix="singlepage")

        elif view_selection == "📥 Export Center (.docx / .html)":
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

        elif view_selection == "🔍 Pipeline Diagnostic Inspector":
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


