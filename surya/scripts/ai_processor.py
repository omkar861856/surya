"""AI Processor for Surya OCR.
Supports OpenAI API and Local AI via Ollama Docker container.
Optimized for HTML & Structure Extraction from OCR Documents.
"""

from __future__ import annotations

import json
import os
import subprocess
import httpx
from typing import List, Tuple, Dict, Any, Generator
import openai


RECOMMENDED_HTML_MODELS = {
    "Ollama (Local)": [
        "qwen2.5",       # Top-tier HTML/XML formatting and multi-lingual layout
        "llama3.2",      # Ultra-fast 3B model for structured extraction & HTML
        "mistral",       # High accuracy 7B reasoning & structure model
        "llava",         # Multimodal Vision-Language Model
    ],
    "OpenAI": [
        "gpt-4o",        # Best-in-class multi-modal & HTML structural reasoning
        "gpt-4o-mini",   # Fast, highly accurate HTML & markdown extractor
        "gpt-4-turbo",
    ]
}


def is_ollama_running() -> bool:
    """Check if Ollama service is reachable on local port 11434."""
    urls = [
        "http://localhost:11434/api/tags",
        "http://127.0.0.1:11434/api/tags",
    ]
    for url in urls:
        try:
            res = httpx.get(url, timeout=2.0)
            if res.status_code == 200:
                return True
        except Exception:
            continue
    return False


def start_ollama_docker() -> Tuple[bool, str]:
    """Ensure Ollama container is running via Docker."""
    try:
        if is_ollama_running():
            return True, "Ollama Docker container is running."

        # Remove conflicting or stopped container named ollama
        subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True, text=True, check=False)
        
        # Launch fresh container
        run_proc = subprocess.run(
            ["docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", "ollama/ollama"],
            capture_output=True,
            text=True,
            check=False,
        )
        if run_proc.returncode == 0:
            return True, "Successfully launched Ollama Docker container."
        else:
            return False, f"Docker run error: {run_proc.stderr}"
    except Exception as e:
        return False, f"Docker command error: {str(e)}"


def get_ollama_models() -> List[str]:
    """Retrieve list of locally available Ollama models."""
    try:
        res = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if res.status_code == 200:
            data = res.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return models
    except Exception:
        pass
    return []


def pull_ollama_model(model_name: str) -> Generator[str, None, None]:
    """Stream pulling an Ollama model from the model registry."""
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name, "stream": True}
    
    try:
        with httpx.stream("POST", url, json=payload, timeout=600.0) as response:
            if response.status_code != 200:
                yield f"Error pulling model: HTTP {response.status_code}"
                return
            
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    status = msg.get("status", "")
                    completed = msg.get("completed")
                    total = msg.get("total")
                    if completed and total:
                        pct = int((completed / total) * 100)
                        yield f"{status} ({pct}%)"
                    elif status:
                        yield status
                except Exception:
                    yield line
    except Exception as e:
        yield f"Pull failed: {str(e)}"


def process_document_with_ai(
    raw_ocr_content: str,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    output_format: str = "html",
    api_key: str | None = None,
) -> str:
    """Process raw OCR output into a structured, presentable HTML/Markdown document optimized for OCR extraction."""
    
    if output_format == "html":
        system_prompt = """You are an expert OCR & Document Intelligence AI Assistant for Tata Power.
Your task is to take raw, unformatted OCR text/HTML extracted from a document page and transform it into clean, semantic, beautifully styled HTML markup.

Strict Requirements:
1. **Semantic HTML**: Use clean tags (`<h1>`, `<h2>`, `<h3>`, `<p>`, `<ul>`, `<li>`, `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`).
2. **Tables**: Convert any table data into proper `<table>` structures with clear header cells `<th>` and data cells `<td>`.
3. **OCR Noise Removal**: Fix typos, transcription glitches, and line wrap breaks without altering numbers, dates, names, or underlying data facts.
4. **Equations**: Keep inline math as \\( ... \\) and display math as \\[ ... \\].
5. **No Code Blocks**: Return ONLY clean HTML code. Do NOT wrap the response in ```html ... ``` code blocks.
"""
    else:
        system_prompt = """You are an expert OCR & Document Intelligence AI Assistant for Tata Power.
Your task is to take raw OCR text/HTML extracted from a document page and transform it into a clean, structured Markdown document.

Strict Requirements:
1. Extract document title (# Title) and section headings (## Heading 2).
2. Correct OCR typos while maintaining 100% data integrity.
3. Structure tables into Markdown tables (`| Col 1 | Col 2 |`).
4. Output ONLY clean Markdown text without code block wrappers.
"""

    user_prompt = f"Transform the following raw OCR document output into a clean, structured document:\n\n{raw_ocr_content}"

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is missing. Please provide your API key in the sidebar.")
        
        client = openai.OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    elif provider == "ollama":
        if not is_ollama_running():
            raise RuntimeError("Ollama is not running on port 11434. Please start the Ollama Docker container.")
        
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")
