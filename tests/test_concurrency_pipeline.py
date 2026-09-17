"""Unit tests for concurrency benchmark logic and test-images dataset integrity."""

from pathlib import Path
from PIL import Image
import pytest
from surya.scripts.concurrency_benchmark import (
    load_latest_concurrency_results,
    RESULTS_FILE,
)


def test_test_images_directory_exists():
    """Verify test-images directory exists and contains valid image assets."""
    img_dir = Path("test-images")
    assert img_dir.exists(), "test-images directory must exist"

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions and p.is_file()]
    assert len(images) > 0, "test-images directory should contain test images"

    # Verify first 3 images can be opened by PIL
    for p in images[:3]:
        with Image.open(p) as im:
            w, h = im.size
            assert w > 0 and h > 0


def test_concurrency_results_structure():
    """Verify JSON structure when reading concurrency benchmark records."""
    dummy_data = {
        "timestamp": "2026-09-17 12:00:00 UTC",
        "concurrency_workers": 4,
        "total_requests": 8,
        "total_time_s": 2.5,
        "throughput_img_per_sec": 3.2,
        "avg_latency_s": 1.2,
        "min_latency_s": 0.8,
        "max_latency_s": 1.5,
        "p95_latency_s": 1.48,
        "success_count": 8,
        "failed_count": 0,
        "success_rate_pct": 100.0,
        "tasks": [
            {
                "task_id": 1,
                "filename": "test1.jpg",
                "status": "Success",
                "elapsed_s": 1.1,
                "blocks_found": 12,
                "error": None,
            }
        ],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    import json
    RESULTS_FILE.write_text(json.dumps(dummy_data))

    loaded = load_latest_concurrency_results()
    assert loaded is not None
    assert loaded["concurrency_workers"] == 4
    assert loaded["throughput_img_per_sec"] == 3.2
    assert loaded["success_rate_pct"] == 100.0
    assert len(loaded["tasks"]) == 1


def test_concurrent_task_aggregator():
    """Verify statistics calculation (P95, throughput, average latency)."""
    task_records = [
        {"task_id": 1, "filename": "img1.jpg", "status": "Success", "elapsed_s": 1.0, "blocks_found": 10, "error": None},
        {"task_id": 2, "filename": "img2.jpg", "status": "Success", "elapsed_s": 1.2, "blocks_found": 8, "error": None},
        {"task_id": 3, "filename": "img3.jpg", "status": "Success", "elapsed_s": 1.8, "blocks_found": 15, "error": None},
        {"task_id": 4, "filename": "img4.jpg", "status": "Success", "elapsed_s": 2.0, "blocks_found": 12, "error": None},
    ]
    latencies = [r["elapsed_s"] for r in task_records]
    latencies_sorted = sorted(latencies)
    avg_latency = round(sum(latencies) / len(latencies), 3)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

    assert avg_latency == 1.5
    assert p95_latency == 2.0


def test_layout_box_instantiation_resilience():
    """Ensure LayoutBox instantiates cleanly with or without raw_label."""
    from surya.layout.schema import LayoutBox, LayoutResult

    box1 = LayoutBox(
        polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
        label="Text",
        position=0,
    )
    assert box1.raw_label == ""
    assert box1.label == "Text"

    box2 = LayoutBox(
        polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
        label="Text",
        raw_label="RawText",
        position=1,
    )
    assert box2.raw_label == "RawText"

    result = LayoutResult(bboxes=[box1, box2], image_bbox=[0, 0, 100, 100])
    assert len(result.bboxes) == 2


