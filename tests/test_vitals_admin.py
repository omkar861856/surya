"""Unit tests for System Vitals & Admin Panel telemetry collection."""

from unittest.mock import patch, MagicMock
from surya.scripts.system_vitals import (
    get_system_vitals,
    get_gpu_vitals,
    get_inference_daemon_vitals,
    get_structurer_daemon_vitals,
    get_service_logs,
)
from surya.scripts.ai_processor import (
    is_local_llm_running,
    analyze_ocr_and_extract_form_fields,
    search_indian_medicine_web,
    search_web_for_indian_medicine,
    extract_medicine_candidates,
)
from surya.scripts.concurrency_benchmark import load_latest_concurrency_results


def test_system_vitals_structure():
    """Verify system vitals returns all expected host hardware metrics."""
    vitals = get_system_vitals()
    assert isinstance(vitals, dict)
    assert "cpu_count" in vitals
    assert "ram_total_mb" in vitals
    assert "ram_used_mb" in vitals
    assert "ram_pct" in vitals
    assert "disk_total_gb" in vitals
    assert "disk_used_gb" in vitals
    assert "disk_pct" in vitals
    assert vitals["cpu_count"] >= 1
    assert vitals["disk_total_gb"] > 0


def test_gpu_vitals_graceful_handling():
    """Verify GPU vitals returns dict or None without unhandled exceptions."""
    vitals = get_gpu_vitals()
    if vitals is not None:
        assert isinstance(vitals, dict)
        assert "name" in vitals
        assert "gpu_utilization_pct" in vitals
        assert "memory_pct" in vitals
        assert "temperature_c" in vitals


def test_inference_daemon_vitals_unreachable():
    """Verify inference daemon vitals reports offline cleanly when port is closed."""
    vitals = get_inference_daemon_vitals(base_url="http://127.0.0.1:59999")
    assert isinstance(vitals, dict)
    assert vitals["healthy"] is False
    assert "Offline" in vitals["status"] or "Unreachable" in vitals["status"]
    assert vitals["slots"] == []


@patch("urllib.request.urlopen")
def test_inference_daemon_vitals_mock_healthy(mock_urlopen):
    """Verify parsing of /health and /slots endpoints when inference server is active."""
    mock_health = MagicMock()
    mock_health.status = 200
    mock_health.__enter__.return_value = mock_health

    mock_slots = MagicMock()
    mock_slots.status = 200
    mock_slots.read.return_value = b'[{"id": 0, "n_ctx": 8192, "is_processing": false, "id_task": 101, "n_prompt_tokens": 50}]'
    mock_slots.__enter__.return_value = mock_slots

    mock_urlopen.side_effect = [mock_health, mock_slots]

    vitals = get_inference_daemon_vitals(base_url="http://127.0.0.1:8000")
    assert vitals["healthy"] is True
    assert vitals["total_slots"] == 1
    assert vitals["active_slots"] == 0
    assert vitals["slots"][0]["id"] == 0
    assert vitals["slots"][0]["prompt_tokens"] == 50


def test_get_service_logs_fallback():
    """Verify service log retriever handles missing services without throwing."""
    logs = get_service_logs("nonexistent-service-12345", lines=10)
    assert isinstance(logs, str)
    assert len(logs) > 0


def test_structurer_daemon_vitals_unreachable():
    """Verify structurer daemon reports offline cleanly when port is closed."""
    vitals = get_structurer_daemon_vitals(base_url="http://127.0.0.1:59998")
    assert isinstance(vitals, dict)
    assert vitals["healthy"] is False


def test_is_local_llm_running_fallback():
    """Verify local LLM check handles closed port gracefully."""
    running = is_local_llm_running(base_url="http://127.0.0.1:59998")
    assert running is False


@patch("openai.resources.chat.completions.Completions.create")
def test_analyze_ocr_and_extract_form_fields_local_llm(mock_create):
    """Verify analyze_ocr_and_extract_form_fields produces expected schema from local LLM."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"document_title": "City Hospital", "document_type": "Medical Prescription", "summary": "Rx for Fever", "is_ai_corrected": true, "ai_corrections_made": ["Normalized Dolo-650"], "dynamic_key_value_fields": [{"field_name": "Patient", "value": "Rahul", "is_medicine_field": false}], "medicines_list": [{"ocr_raw_name": "Dolo", "corrected_medicine_name": "Dolo 650mg", "composition": "Paracetamol", "dosage": "1-0-1", "duration": "5 days", "correction_status": "Verified"}], "content_sections": []}'
    mock_resp.choices = [mock_choice]
    mock_create.return_value = mock_resp

    data = analyze_ocr_and_extract_form_fields(
        raw_ocr_content="Dr Sharma Patient Rahul Dolo 650",
        model_name="qwen2.5-7b-instruct",
        base_url="http://127.0.0.1:8001/v1",
    )
    assert data["document_title"] == "City Hospital"
    assert data["is_ai_corrected"] is True
    assert len(data["medicines_list"]) == 1
    assert data["medicines_list"][0]["corrected_medicine_name"] == "Dolo 650mg"
    assert len(data["dynamic_key_value_fields"]) == 1
    assert "search_grounding_metadata" in data
    assert "queries_searched" in data["search_grounding_metadata"]


def test_extract_medicine_candidates():
    """Verify smart clinical candidate extraction filters stop words and identifies medicines."""
    sample_text = """
    CITY CARE CLINIC
    Dr. Sharma, MBBS
    Patient: Rahul Gupta, Age: 35
    Rx:
    1. Tab Dolo 650mg 1-0-1 after food x 3 days
    2. Cap Pan-D 1-0-0 before breakfast x 5 days
    3. Augmentin 625 Duo 1-0-1
    """
    cands = extract_medicine_candidates(sample_text)
    assert isinstance(cands, list)
    assert any("Dolo" in c for c in cands)
    assert any("Pan-D" in c for c in cands)
    assert any("Augmentin" in c for c in cands)
    # Stop words must be filtered out
    assert not any(c.lower() in ("patient", "doctor", "sharma", "clinic") for c in cands)


def test_search_indian_medicine_web():
    """Verify search_indian_medicine_web resolves Indian brand names and chemical compositions."""
    res = search_indian_medicine_web("Dolo 650")
    assert isinstance(res, dict)
    assert res.get("found") is True
    assert "Dolo" in res.get("brand_name", "")
    assert "Paracetamol" in res.get("composition", "")
    assert res.get("manufacturer") != ""

    snippet = search_web_for_indian_medicine("Augmentin 625")
    assert isinstance(snippet, str)
    assert "Augmentin" in snippet


def test_search_indian_medicine_web_fallback():
    """Verify search_indian_medicine_web handles empty or unknown inputs safely."""
    empty_res = search_indian_medicine_web("")
    assert empty_res.get("found") is False

    short_res = search_indian_medicine_web("x")
    assert short_res.get("found") is False

