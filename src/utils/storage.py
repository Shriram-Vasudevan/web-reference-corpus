"""SQLite storage layer for website style classification."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import config


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a connection with row factory enabled."""
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Context manager that commits on success, rolls back on error."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(conn: sqlite3.Connection | None = None):
    """Create all tables if they don't exist."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sites (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT UNIQUE NOT NULL,
            domain      TEXT NOT NULL,
            category_hint TEXT,
            screenshot_path TEXT,
            captured_at TIMESTAMP,
            status      TEXT DEFAULT 'pending',
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            site_id     INTEGER PRIMARY KEY REFERENCES sites(id),
            vector      BLOB NOT NULL,
            model_name  TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS umap_coords (
            site_id     INTEGER PRIMARY KEY REFERENCES sites(id),
            run_id      TEXT NOT NULL,
            x_20d       BLOB,
            x_2d_0      REAL,
            x_2d_1      REAL
        );

        CREATE TABLE IF NOT EXISTS clusters (
            site_id     INTEGER NOT NULL REFERENCES sites(id),
            run_id      TEXT NOT NULL,
            cluster_id  INTEGER NOT NULL,
            probability REAL,
            PRIMARY KEY (site_id, run_id)
        );

        CREATE TABLE IF NOT EXISTS style_labels (
            cluster_id          INTEGER NOT NULL,
            run_id              TEXT NOT NULL,
            page_type           TEXT NOT NULL,
            visual_style        TEXT NOT NULL,
            quality_score       INTEGER NOT NULL,
            industry            TEXT,
            color_mode          TEXT,
            layout_pattern      TEXT,
            typography_style    TEXT,
            design_era          TEXT,
            target_audience     TEXT,
            distinguishing_features TEXT,
            raw_response        TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (cluster_id, run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_sites_status ON sites(status);
        CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id, cluster_id);
    """)

    if close:
        conn.close()


# ── Site helpers ───────────────────────────────────────────────────────

def upsert_site(conn: sqlite3.Connection, url: str, domain: str,
                category_hint: str | None = None) -> int:
    """Insert or update a site, returning its id."""
    conn.execute(
        """INSERT INTO sites (url, domain, category_hint)
           VALUES (?, ?, ?)
           ON CONFLICT(url) DO UPDATE SET category_hint=excluded.category_hint""",
        (url, domain, category_hint),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM sites WHERE url=?", (url,)).fetchone()
    return row["id"]


def mark_captured(conn: sqlite3.Connection, site_id: int, path: str):
    """Mark a site as captured with its screenshot path."""
    conn.execute(
        "UPDATE sites SET screenshot_path=?, status='captured', captured_at=CURRENT_TIMESTAMP WHERE id=?",
        (path, site_id),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, site_id: int, error: str):
    """Mark a site as failed with error detail."""
    conn.execute(
        "UPDATE sites SET status='failed', error=? WHERE id=?",
        (error, site_id),
    )
    conn.commit()


def get_captured_sites(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all successfully captured sites."""
    return conn.execute(
        "SELECT * FROM sites WHERE status='captured' ORDER BY id"
    ).fetchall()


def get_pending_sites(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return sites not yet captured."""
    return conn.execute(
        "SELECT * FROM sites WHERE status='pending' ORDER BY id"
    ).fetchall()


# ── Embedding helpers ──────────────────────────────────────────────────

def store_embedding(conn: sqlite3.Connection, site_id: int,
                    vector: np.ndarray, model_name: str):
    """Store a single embedding vector."""
    conn.execute(
        """INSERT INTO embeddings (site_id, vector, model_name)
           VALUES (?, ?, ?)
           ON CONFLICT(site_id) DO UPDATE SET vector=excluded.vector,
           model_name=excluded.model_name, created_at=CURRENT_TIMESTAMP""",
        (site_id, vector.tobytes(), model_name),
    )
    conn.commit()


def get_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Return (site_ids, embedding_matrix) for all stored embeddings."""
    rows = conn.execute(
        "SELECT site_id, vector FROM embeddings ORDER BY site_id"
    ).fetchall()
    if not rows:
        return [], np.array([])
    site_ids = [r["site_id"] for r in rows]
    vectors = np.stack([
        np.frombuffer(r["vector"], dtype=np.float32) for r in rows
    ])
    return site_ids, vectors


def get_embedding(conn: sqlite3.Connection, site_id: int) -> np.ndarray | None:
    """Return the embedding vector for a single site."""
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE site_id=?", (site_id,)
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32)


# ── Cluster helpers ────────────────────────────────────────────────────

def store_clusters(conn: sqlite3.Connection, run_id: str,
                   site_ids: list[int], labels: np.ndarray,
                   probabilities: np.ndarray):
    """Store cluster assignments for a run."""
    with transaction(conn):
        conn.execute("DELETE FROM clusters WHERE run_id=?", (run_id,))
        conn.executemany(
            "INSERT INTO clusters (site_id, run_id, cluster_id, probability) VALUES (?,?,?,?)",
            [(int(sid), run_id, int(lbl), float(prob))
             for sid, lbl, prob in zip(site_ids, labels, probabilities)],
        )


def get_cluster_members(conn: sqlite3.Connection, run_id: str,
                        cluster_id: int) -> list[sqlite3.Row]:
    """Return sites in a specific cluster."""
    return conn.execute(
        """SELECT s.*, c.probability FROM clusters c
           JOIN sites s ON s.id = c.site_id
           WHERE c.run_id=? AND c.cluster_id=?
           ORDER BY c.probability DESC""",
        (run_id, cluster_id),
    ).fetchall()


def get_cluster_ids(conn: sqlite3.Connection, run_id: str) -> list[int]:
    """Return distinct cluster IDs for a run (excluding noise=-1)."""
    rows = conn.execute(
        "SELECT DISTINCT cluster_id FROM clusters WHERE run_id=? AND cluster_id >= 0 ORDER BY cluster_id",
        (run_id,),
    ).fetchall()
    return [r["cluster_id"] for r in rows]


def get_site_cluster(conn: sqlite3.Connection, run_id: str,
                     site_id: int) -> int | None:
    """Return cluster_id for a site in a given run."""
    row = conn.execute(
        "SELECT cluster_id FROM clusters WHERE run_id=? AND site_id=?",
        (run_id, site_id),
    ).fetchone()
    return row["cluster_id"] if row else None


# ── Style label helpers ────────────────────────────────────────────────

def store_style_label(conn: sqlite3.Connection, cluster_id: int, run_id: str,
                      label_data: dict, raw_response: str):
    """Store a style label for a cluster."""
    conn.execute(
        """INSERT INTO style_labels
           (cluster_id, run_id, page_type, visual_style, quality_score,
            industry, color_mode, layout_pattern, typography_style,
            design_era, target_audience, distinguishing_features, raw_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(cluster_id, run_id) DO UPDATE SET
            page_type=excluded.page_type,
            visual_style=excluded.visual_style,
            quality_score=excluded.quality_score,
            industry=excluded.industry,
            color_mode=excluded.color_mode,
            layout_pattern=excluded.layout_pattern,
            typography_style=excluded.typography_style,
            design_era=excluded.design_era,
            target_audience=excluded.target_audience,
            distinguishing_features=excluded.distinguishing_features,
            raw_response=excluded.raw_response,
            created_at=CURRENT_TIMESTAMP""",
        (
            cluster_id, run_id,
            label_data.get("page_type", ""),
            label_data.get("visual_style", ""),
            label_data.get("quality_score", 0),
            label_data.get("industry", ""),
            label_data.get("color_mode", ""),
            label_data.get("layout_pattern", ""),
            label_data.get("typography_style", ""),
            label_data.get("design_era", ""),
            label_data.get("target_audience", ""),
            label_data.get("distinguishing_features", ""),
            raw_response,
        ),
    )
    conn.commit()


def get_style_label(conn: sqlite3.Connection, run_id: str,
                    cluster_id: int) -> sqlite3.Row | None:
    """Return the style label for a cluster in a given run."""
    return conn.execute(
        "SELECT * FROM style_labels WHERE run_id=? AND cluster_id=?",
        (run_id, cluster_id),
    ).fetchone()


def get_all_style_labels(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """Return all style labels for a run."""
    return conn.execute(
        "SELECT * FROM style_labels WHERE run_id=? ORDER BY cluster_id",
        (run_id,),
    ).fetchall()


# ── UMAP coord helpers ─────────────────────────────────────────────────

def store_umap_coords(conn: sqlite3.Connection, run_id: str,
                      site_ids: list[int], coords_2d: np.ndarray) -> None:
    """Persist 2D UMAP visualization coordinates for each site."""
    with transaction(conn):
        conn.executemany(
            """INSERT INTO umap_coords (site_id, run_id, x_2d_0, x_2d_1)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(site_id) DO UPDATE SET
                 run_id=excluded.run_id,
                 x_2d_0=excluded.x_2d_0,
                 x_2d_1=excluded.x_2d_1""",
            [
                (int(sid), run_id, float(coords_2d[i, 0]), float(coords_2d[i, 1]))
                for i, sid in enumerate(site_ids)
            ],
        )


def get_umap_coords(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all stored 2D UMAP coordinates (most recent run per site)."""
    return conn.execute(
        "SELECT site_id, x_2d_0, x_2d_1 FROM umap_coords"
    ).fetchall()


def get_latest_run_id(conn: sqlite3.Connection) -> str | None:
    """Return the most recent run_id from clusters table."""
    row = conn.execute(
        "SELECT run_id FROM clusters ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def get_site_by_url(conn: sqlite3.Connection, url: str) -> sqlite3.Row | None:
    """Look up a site by URL."""
    return conn.execute("SELECT * FROM sites WHERE url=?", (url,)).fetchone()
