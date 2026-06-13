# Asset Intel

IT asset identification via a two-stage computer vision pipeline. A warehouse worker photographs hardware; the system returns the asset class, YOLO confidence score, and CLIP similarity score — no manual entry required.

> **Status:** Active development — v1.0.0. Validated on local hardware. Not yet load-tested for multi-tenant production use.

---

## How It Works

1. Worker uploads a photo via the Streamlit UI or calls `POST /scan` directly
2. YOLOv8m detects and localizes all hardware items in the frame
3. Each detected crop is encoded by CLIP ViT-B/32 into a 512-dimensional vector
4. FAISS finds the nearest catalog match; ChromaDB returns the associated metadata
5. Results and a full audit record are written to SQLite (PostgreSQL-ready)
6. Redis caches results — repeated scans on the same image are served instantly

Multiple items in a single photo are handled in one request.

---

## Model Performance

| Model | Training Set | Epochs | mAP50 | mAP50-95 |
|-------|-------------|--------|-------|----------|
| YOLOv8m | 1,000 images (Open Images V7) | 50 | 0.510 | 0.285 |

**Detected categories:** Laptop, Computer monitor, Computer keyboard, Printer, Telephone, Television

mAP50 of 0.510 is acceptable for a first-pass classifier on a small training set. Performance improves substantially with a larger dataset — the retraining pipeline is designed for this (see [MLOps](#mlops)).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Open Images V7, FiftyOne, Albumentations, DVC |
| Detection | YOLOv8m (Ultralytics 8.1.0) |
| Embeddings | CLIP ViT-B/32 (open-clip-torch) |
| Vector Search | FAISS IndexFlatIP, ChromaDB |
| MLOps | MLflow 2.11, Evidently, Prefect, DVC |
| Backend | FastAPI 0.110, SQLAlchemy, Redis, SQLite |
| UI | Streamlit |
| DevOps | Docker, Docker Compose |

---

## Prerequisites

- Python 3.11
- Conda or venv
- Git
- NVIDIA GPU with CUDA 12.1 (CPU inference works but is significantly slower)
- Redis (optional — falls back to in-memory cache if unavailable)

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/Harkishan-Singh-Gabri/ASSET-INTEL.git
cd ASSET-INTEL
```

### 2. Environment

```bash
conda create -n asset-intel python=3.11 -y
conda activate asset-intel
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env — see Environment Variables below
```

### 4. Data Pipeline

```bash
python src/data/download.py       # ~20 min — downloads 1,000 images from Open Images V7
python src/data/augment.py        # applies warehouse-condition augmentations
python src/data/prepare_yolo.py   # converts COCO format to YOLO, filters categories
```

### 5. Training

```bash
python src/models/train_yolo.py   # ~15 min on GPU; logs to MLflow
python src/models/build_index.py  # builds FAISS index + ChromaDB metadata store
```

### 6. Start Services

```bash
# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
streamlit run src/UI/app.py

# Terminal 3 (optional)
mlflow ui --host 127.0.0.1 --port 5000
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `sqlite:///./assetintel.db` | SQLAlchemy connection string |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection; omit to use in-memory fallback |
| `YOLO_WEIGHTS` | Yes | — | Path to trained `.pt` weights file |
| `FAISS_INDEX` | Yes | — | Path to `.faiss` index file |
| `CHROMA_DIR` | Yes | — | Path to ChromaDB directory |
| `CONFIDENCE_THRESHOLD` | No | `0.65` | CLIP score below which a scan is auto-flagged |
| `FLAG_RATE_THRESHOLD` | No | `0.40` | Flag rate above which drift alert is raised |

---

## API Reference

### `POST /scan`

Runs the full two-stage pipeline on an uploaded image.

**Request:** `multipart/form-data` with an image file field named `file`

**Response:**
```json
{
  "scan_id": "uuid",
  "results": [
    {
      "label": "Computer monitor",
      "yolo_confidence": 0.97,
      "clip_score": 0.849,
      "flagged": false,
      "bounding_box": [x1, y1, x2, y2]
    }
  ],
  "cached": false
}
```

### `GET /scans`

Returns the full audit log of all scans, ordered by timestamp descending.

---

## Data Flow

```
Worker uploads photo
        |
POST /scan (FastAPI)
        |
Redis cache check --> hit: return cached result
        |
        v (cache miss)
YOLOv8m -- detect and localize hardware items
        |
CLIP ViT-B/32 -- encode each crop to 512-dim vector
        |
FAISS IndexFlatIP -- top-5 nearest neighbors
        |
ChromaDB -- retrieve metadata for matches
        |
SQLite -- write audit record
        |
Streamlit -- display results
```

---

## Project Structure

```
ASSET-INTEL/
├── assets/
├── data/
│   ├── raw/                    # DVC-tracked — Open Images V7
│   ├── processed/              # DVC-tracked — augmented images
│   ├── yolo/                   # DVC-tracked — YOLO format dataset
│   └── embeddings/
│       ├── assets.faiss
│       └── chroma/
├── models/
│   └── yolo/v1/weights/
│       └── best.pt             # DVC-tracked
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── augment.py
│   │   └── prepare_yolo.py
│   ├── models/
│   │   ├── train_yolo.py
│   │   └── build_index.py
│   ├── inference/
│   │   ├── pipeline.py
│   │   └── vector_search.py
│   ├── api/
│   │   ├── main.py
│   │   ├── db.py
│   │   └── cache.py
│   └── UI/
│       └── app.py
├── mlops/
│   ├── drift_detection.py
│   ├── retrain_flow.py
│   └── reports/
├── docker/
│   ├── docker-compose.yml
│   └── api.Dockerfile
├── dvc.yaml
├── .env.example
└── requirements.txt
```

---

## MLOps

### Experiment Tracking

All training runs log mAP50, mAP50-95, Precision, Recall, and model weights automatically.

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

### Drift Detection

Raises an alert when:
- Average CLIP score falls below `0.65` — model struggling with unseen hardware types
- Flag rate exceeds `40%` — too many low-confidence matches in production

```bash
python mlops/drift_detection.py
# Output: JSON report written to mlops/reports/
```

### Automated Retraining

```bash
# Trigger manually (runs only if drift is detected)
python -m mlops.retrain_flow

# Force retraining regardless of drift status
python -m mlops.retrain_flow --force
```

Pipeline stages: `download → augment → prepare → train → rebuild index`

### Data Versioning

```bash
dvc repro       # reproduce full pipeline from scratch
dvc commit      # snapshot current data state
dvc push        # push to remote storage
dvc pull        # restore data on a new machine
```

---

## Docker

```bash
cd docker
docker compose up -d
```

To use PostgreSQL instead of SQLite, update `.env`:

```dotenv
DATABASE_URL=postgresql://user:pass@postgres:5432/assetintel
```

---

## Known Limitations

- Training set of 1,000 images is small. mAP50 improves significantly with 5,000+ images. Use `retrain_flow.py` to expand the dataset.
- Only six hardware categories are supported out of the box. Adding a new category requires retraining YOLO and rebuilding the FAISS index.
- SQLite is suitable for single-instance use. Switch to PostgreSQL via `DATABASE_URL` for concurrent write workloads.
- The API has no authentication layer. Do not expose port 8000 to untrusted networks without adding one.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run the full pipeline and confirm `mAP50 >= 0.45` before submitting
4. Test with both Redis available and unavailable (in-memory fallback path)
5. Open a pull request against `main`

---

## License

MIT — see [LICENSE](LICENSE).

---

**Author:** [Harkishan Singh Gabri](https://github.com/Harkishan-Singh-Gabri) — [LinkedIn](https://www.linkedin.com/in/harkishan-singh-gabri/)
