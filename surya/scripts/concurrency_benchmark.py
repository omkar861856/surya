"""Concurrency Benchmark Runner for Surya OCR.
Tests simultaneous multi-user document processing using images from test-images/.
Measures throughput (img/s), average latency, P95 latency, slot contention, and error rates.
Saves results to output/concurrency_results.json for the Admin Panel.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

RESULTS_FILE = Path("output/concurrency_results.json")


def load_latest_concurrency_results() -> Optional[Dict[str, Any]]:
    """Read the latest benchmark results from JSON file."""
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    return None


def run_concurrency_test(
    image_dir: str = "test-images",
    concurrency: int = 4,
    total_images: int = 8,
    backend_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute concurrent OCR requests across multiple threads."""
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Gather available test images
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in valid_extensions and p.is_file()
    ]
    if not image_paths:
        raise ValueError(f"No valid images found in {image_dir}")

    # Select target subset
    selected_paths = image_paths[:total_images]
    if len(selected_paths) < total_images:
        # Cycle if fewer files
        selected_paths = (selected_paths * ((total_images // len(selected_paths)) + 1))[:total_images]

    # Initialize predictor
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor

    manager = SuryaInferenceManager()
    rec_predictor = RecognitionPredictor(manager)

    task_records: List[Dict[str, Any]] = []

    def _process_one(idx: int, path: Path) -> Dict[str, Any]:
        t0 = time.perf_counter()
        record: Dict[str, Any] = {
            "task_id": idx + 1,
            "filename": path.name,
            "status": "Failed",
            "elapsed_s": 0.0,
            "blocks_found": 0,
            "error": None,
        }
        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                results = rec_predictor([rgb_img], full_page=True)
                page = results[0]
                record["blocks_found"] = len(page.blocks)
                record["status"] = "Success"
        except Exception as e:
            record["error"] = str(e)
        finally:
            record["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return record

    t_bench_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_process_one, i, p)
            for i, p in enumerate(selected_paths)
        ]
        for f in concurrent.futures.as_completed(futures):
            task_records.append(f.result())

    total_bench_time = round(time.perf_counter() - t_bench_start, 3)

    # Sort records by task_id
    task_records.sort(key=lambda x: x["task_id"])

    # Aggregate Statistics
    successful = [r for r in task_records if r["status"] == "Success"]
    success_count = len(successful)
    fail_count = len(task_records) - success_count
    latencies = [r["elapsed_s"] for r in task_records]

    latencies_sorted = sorted(latencies)
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    min_latency = round(min(latencies), 3) if latencies else 0.0
    max_latency = round(max(latencies), 3) if latencies else 0.0
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95_latency = round(latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)], 3) if latencies else 0.0
    throughput = round(len(task_records) / total_bench_time, 2) if total_bench_time > 0 else 0.0
    success_rate = round((success_count / len(task_records)) * 100.0, 1) if task_records else 0.0

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "concurrency_workers": concurrency,
        "total_requests": len(task_records),
        "total_time_s": total_bench_time,
        "throughput_img_per_sec": throughput,
        "avg_latency_s": avg_latency,
        "min_latency_s": min_latency,
        "max_latency_s": max_latency,
        "p95_latency_s": p95_latency,
        "success_count": success_count,
        "failed_count": fail_count,
        "success_rate_pct": success_rate,
        "tasks": task_records,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Surya OCR concurrency test")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--count", type=int, default=8, help="Total number of images to process")
    parser.add_argument("--dir", type=str, default="test-images", help="Test images directory")
    args = parser.parse_args()

    print(f"Starting concurrency benchmark: {args.concurrency} workers, {args.count} requests from {args.dir}...")
    res = run_concurrency_test(image_dir=args.dir, concurrency=args.concurrency, total_images=args.count)
    print("\n--- Benchmark Results ---")
    print(f"Throughput:    {res['throughput_img_per_sec']} img/s")
    print(f"Total Time:    {res['total_time_s']}s")
    print(f"Avg Latency:   {res['avg_latency_s']}s")
    print(f"P95 Latency:   {res['p95_latency_s']}s")
    print(f"Success Rate:  {res['success_rate_pct']}% ({res['success_count']}/{res['total_requests']})")
