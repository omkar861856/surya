# Enterprise Scalability & Deployment Guide

This guide walks through configuring each of the 7 cloud infrastructure components to eliminate **502 Bad Gateway** errors and scale Surya OCR to handle multiple simultaneous users.

---

## 1. Images (Containerization)

1. **Build the container image**:
   ```bash
   docker build -t <your-registry>/suryaocr:latest .
   ```
2. **Push to your Cloud Container Registry**:
   ```bash
   docker push <your-registry>/suryaocr:latest
   ```
3. Update `k8s/frontend-deployment.yaml` with your image URL (`<your-registry>/suryaocr:latest`).

---

## 2. Nodes (Compute Fleet)

In your cloud provider dashboard:
- **Node Pool 1 (Frontend Web)**:
  - Role: Runs the auto-scaling Streamlit frontend pods.
  - Instance Type: Standard CPU nodes (e.g., 2-4 vCPU, 8-16 GB RAM).
  - Count: 2 to 5 nodes.
- **Node Pool 2 (GPU Inference)**:
  - Role: Runs the persistent `llama-ocr` model server.
  - Instance Type: 1x NVIDIA RTX 4090 / 5090 / A10G / L4 (16GB - 32GB VRAM).
  - Node Label: `accelerator=nvidia-gpu`.
  - Taint: `nvidia.com/gpu=present:NoSchedule` (ensures CPU-only pods don't waste GPU capacity).

---

## 3. Kubernetes (Cluster Orchestration)

Deploy the entire architecture with one command:
```bash
kubectl apply -k k8s/
```

Verify the deployment:
```bash
kubectl get pods -n surya-ocr -o wide
```

### Self-Healing & Zero 502s:
* **Startup / Readiness / Liveness Probes**: Continuously check `/_stcore/health`.
* If a heavy PDF upload freezes a pod, Kubernetes isolates that pod, diverts all user traffic to surviving healthy pods, and restarts the frozen pod automatically.

---

## 4. Load Balancers (Traffic Distribution & Session Stickiness)

In your cloud dashboard or via the provided `k8s/ingress-loadbalancer.yaml`:
* **Session Affinity (Cookie-based Sticky Sessions)**:
  * **Status**: **MANDATORY**.
  * **Why**: Streamlit maintains persistent WebSockets (`/_stcore/stream`). Without sticky sessions, each new WebSocket packet hits a different pod, causing connection drops and bad gateway errors.
* **WebSocket Support**: Ensure HTTP Upgrade / WebSocket proxying is enabled.
* **Timeouts**: Increase proxy read and write timeouts to **3600 seconds** for long-running OCR jobs.
* **Health Check**: Path `/_stcore/health`, port `8501`, timeout `5s`.

---

## 5. Auto Scaling (Dynamic Capacity)

Configured in `k8s/frontend-hpa.yaml`:
* **Min Replicas**: 2
* **Max Replicas**: 10
* **Metrics**:
  * Scale up when average CPU exceeds **70%**.
  * Scale up when average Memory exceeds **75%**.
* **Cluster Autoscaler**: In your cloud dashboard, enable cluster autoscaling on Node Pool 1 (min: 2 nodes, max: 8 nodes). When HPA creates more pods than current nodes can fit, new compute nodes spin up automatically in 60-90 seconds.

---

## 6. Volumes (Persistent Storage)

Configured in `k8s/volumes-pvc.yaml`:
* **`surya-model-cache-pvc` (50 GiB)**:
  * Mount path: `/root/.cache/huggingface` and `/models`.
  * **Benefit**: The 1.8 GB GGUF model weights and MMProj weights are saved permanently on this volume. When new pods scale up or restart, they load the weights in milliseconds without re-downloading from Hugging Face.
* **`surya-data-pvc` (20 GiB)**:
  * Storage for output files, export zip archives, and cached document hashes.

---

## 7. CDP Backups (Continuous Data Protection)

In your cloud dashboard under **Volumes / Backups**:
* Enable **CDP (Continuous Data Protection)** or **Automated Scheduled Snapshots** for `surya-model-cache-pvc` and `surya-data-pvc`.
* Recommended Schedule: Daily snapshot with 7-day retention.
* Point-in-time recovery allows rolling back data within seconds if files or configurations become corrupted.
