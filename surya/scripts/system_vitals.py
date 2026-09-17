"""System and GPU Vitals Monitor for Surya OCR Admin Panel.
Collects real-time telemetry from NVIDIA-SMI, host RAM, CPU load, disk storage,
and the llama-ocr inference engine slots without requiring third-party C libraries.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error


def get_gpu_vitals() -> Optional[Dict[str, Any]]:
    """Query NVIDIA-SMI for real-time GPU statistics."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode != 0 or not res.stdout.strip():
            return None

        parts = [p.strip() for p in res.stdout.strip().split(",")]
        if len(parts) >= 7:
            name = parts[0]
            driver = parts[1]
            gpu_util = float(parts[2])
            mem_used = float(parts[3])
            mem_total = float(parts[4])
            temp = float(parts[5])
            power = float(parts[6])
            mem_pct = round((mem_used / mem_total) * 100, 1) if mem_total > 0 else 0.0

            return {
                "available": True,
                "name": name,
                "driver": driver,
                "gpu_utilization_pct": gpu_util,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_pct": mem_pct,
                "temperature_c": temp,
                "power_draw_w": power,
            }
    except Exception:
        pass
    return None


def get_system_vitals() -> Dict[str, Any]:
    """Collect host CPU load, RAM usage, and disk storage."""
    vitals: Dict[str, Any] = {
        "load_1m": 0.0,
        "load_5m": 0.0,
        "load_15m": 0.0,
        "ram_used_mb": 0.0,
        "ram_total_mb": 0.0,
        "ram_pct": 0.0,
        "disk_used_gb": 0.0,
        "disk_total_gb": 0.0,
        "disk_pct": 0.0,
        "cpu_count": os.cpu_count() or 1,
    }

    # Load average
    try:
        l1, l5, l15 = os.getloadavg()
        vitals["load_1m"] = round(l1, 2)
        vitals["load_5m"] = round(l5, 2)
        vitals["load_15m"] = round(l15, 2)
    except Exception:
        pass

    # RAM from /proc/meminfo or free -m
    try:
        mem_info: Dict[str, float] = {}
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem_info[key] = float(val) / 1024.0  # Convert kB to MB
            total = mem_info.get("MemTotal", 0.0)
            avail = mem_info.get("MemAvailable", mem_info.get("MemFree", 0.0))
            used = total - avail
            vitals["ram_total_mb"] = round(total, 1)
            vitals["ram_used_mb"] = round(used, 1)
            vitals["ram_pct"] = round((used / total) * 100, 1) if total > 0 else 0.0
    except Exception:
        pass

    # Disk usage
    try:
        total_b, used_b, free_b = shutil.disk_usage("/")
        total_gb = round(total_b / (1024**3), 1)
        used_gb = round(used_b / (1024**3), 1)
        vitals["disk_total_gb"] = total_gb
        vitals["disk_used_gb"] = used_gb
        vitals["disk_pct"] = round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0
    except Exception:
        pass

    return vitals


def get_inference_daemon_vitals(base_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    """Ping llama-ocr inference daemon and collect slot allocation status."""
    vitals: Dict[str, Any] = {
        "status": "Offline / Unreachable",
        "healthy": False,
        "latency_ms": None,
        "slots": [],
        "active_slots": 0,
        "total_slots": 0,
    }

    # 1. Ping /health
    health_url = f"{base_url}/health"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "SuryaAdmin/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            latency = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200:
                vitals["status"] = "Healthy & Running"
                vitals["healthy"] = True
                vitals["latency_ms"] = round(latency, 1)
    except Exception as e:
        vitals["status"] = f"Unreachable ({type(e).__name__})"
        return vitals

    # 2. Query /slots
    slots_url = f"{base_url}/slots"
    try:
        req = urllib.request.Request(slots_url, headers={"User-Agent": "SuryaAdmin/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                slots = []
                active_count = 0
                for item in data:
                    is_proc = item.get("is_processing", False)
                    if is_proc:
                        active_count += 1
                    slots.append({
                        "id": item.get("id"),
                        "n_ctx": item.get("n_ctx", 8192),
                        "is_processing": is_proc,
                        "task_id": item.get("id_task", "-"),
                        "prompt_tokens": item.get("n_prompt_tokens", 0),
                    })
                vitals["slots"] = slots
                vitals["total_slots"] = len(slots)
                vitals["active_slots"] = active_count
    except Exception:
        pass

    return vitals


def get_structurer_daemon_vitals(base_url: str = "http://127.0.0.1:8001") -> Dict[str, Any]:
    """Ping llama-structurer (Qwen2.5-7B on port 8001) and collect slot allocation status."""
    return get_inference_daemon_vitals(base_url=base_url)


def restart_inference_service(service_name: str = "llama-ocr") -> tuple[bool, str]:
    """Restart llama-ocr or llama-structurer systemd service."""
    try:
        cmd = ["sudo", "systemctl", "restart", service_name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            return True, f"Service '{service_name}' restarted successfully!"
        return False, f"Failed: {res.stderr or res.stdout}"
    except Exception as e:
        return False, f"Error: {e}"


def get_service_logs(service_name: str = "llama-ocr", lines: int = 40) -> str:
    """Retrieve the most recent log output for a systemd service."""
    try:
        cmd = ["journalctl", f"-u", service_name, "-n", str(lines), "--no-pager"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
        return f"No logs available or service not active: {res.stderr or res.stdout}"
    except Exception as e:
        return f"Could not retrieve logs: {e}"

