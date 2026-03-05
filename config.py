"""Central configuration for the Website Style Classification system."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
DB_PATH = DATA_DIR / "website_styles.db"
EMBEDDINGS_PATH = DATA_DIR / "all_embeddings.npy"
UMAP_20D_PATH = DATA_DIR / "umap_20d.npy"
UMAP_2D_PATH = DATA_DIR / "umap_2d.npy"
OUTPUTS_DIR = ROOT / "outputs"
SEEDS_PATH = ROOT / "seeds.csv"

# ── Screenshot capture ─────────────────────────────────────────────────
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900
SCREENSHOT_TIMEOUT_MS = 30_000
SETTLE_TIME_S = 3
MAX_CONCURRENT_BROWSERS = 3
CAPTURE_RETRIES = 2

# ── CLIP model ─────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 512
EMBED_BATCH_SIZE = 16

# ── UMAP ───────────────────────────────────────────────────────────────
UMAP_N_COMPONENTS_CLUSTER = 20
UMAP_N_COMPONENTS_VIZ = 2
UMAP_METRIC = "cosine"
UMAP_MIN_DIST_CLUSTER = 0.0
UMAP_MIN_DIST_VIZ = 0.1
UMAP_N_NEIGHBORS = 15

# ── HDBSCAN ────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_CLUSTER_SELECTION = "eom"
NOISE_WARN_THRESHOLD = 0.30

# ── Claude VLM labeling ───────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
LABEL_SAMPLES_PER_CLUSTER = 5
LABEL_MAX_RETRIES = 3

# ── Retrieval ──────────────────────────────────────────────────────────
DEFAULT_TOP_K = 10

# Ensure directories exist
for d in [DATA_DIR, SCREENSHOTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
