#!/usr/bin/env python3
"""Step 6: Export an interactive HTML atlas of the website style landscape."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import (
    get_connection,
    get_all_embeddings,
    get_all_style_labels,
    get_cluster_ids,
    get_cluster_members,
    get_latest_run_id,
    get_umap_coords,
    init_db,
)

console = Console()

# Generous palette so every cluster gets a distinct color
_PALETTE = [
    "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316",
    "#eab308", "#22c55e", "#10b981", "#06b6d4", "#3b82f6",
    "#a855f7", "#d946ef", "#ef4444", "#f59e0b", "#84cc16",
    "#14b8a6", "#0ea5e9", "#6d28d9", "#be185d", "#047857",
    "#1d4ed8", "#b45309", "#7c3aed", "#db2777", "#0f766e",
]
_NOISE_COLOR = "rgba(120, 120, 140, 0.35)"


# ── Data loading ───────────────────────────────────────────────────────

def _load_coords(conn) -> dict[int, tuple[float, float]]:
    """Return {site_id: (x, y)} from DB or fall back to .npy file."""
    rows = get_umap_coords(conn)
    if rows:
        return {r["site_id"]: (r["x_2d_0"], r["x_2d_1"]) for r in rows}

    if not config.UMAP_2D_PATH.exists():
        return {}

    umap_2d = np.load(config.UMAP_2D_PATH)
    embed_ids, _ = get_all_embeddings(conn)
    if len(embed_ids) != len(umap_2d):
        console.print(
            f"[yellow]Warning: embedding count ({len(embed_ids)}) ≠ "
            f"UMAP rows ({len(umap_2d)}). Coords may be misaligned.[/]"
        )
        n = min(len(embed_ids), len(umap_2d))
        embed_ids, umap_2d = embed_ids[:n], umap_2d[:n]

    return {
        int(sid): (float(umap_2d[i, 0]), float(umap_2d[i, 1]))
        for i, sid in enumerate(embed_ids)
    }


def build_dataframe(conn, run_id: str) -> pd.DataFrame | None:
    """Join sites, coords, clusters, and style labels into a flat DataFrame."""
    coords_map = _load_coords(conn)
    if not coords_map:
        console.print("[red]No UMAP coordinates found. Run 03_cluster.py first.[/]")
        return None

    label_map = {
        lbl["cluster_id"]: dict(lbl)
        for lbl in get_all_style_labels(conn, run_id)
    }
    cluster_ids = get_cluster_ids(conn, run_id)

    rows: list[dict] = []
    for cid in [-1] + cluster_ids:
        members = get_cluster_members(conn, run_id, cid)
        lbl = label_map.get(cid, {})
        style_name = (
            lbl.get("umbrella_label") or
            ("Noise / Outliers" if cid == -1 else f"Cluster {cid}")
        )

        for m in members:
            sid = m["id"]
            if sid not in coords_map:
                continue
            x, y = coords_map[sid]

            # Build a relative path for the screenshot preview (works when
            # website_atlas.html is opened from the outputs/ directory).
            shot = m["screenshot_path"] or ""
            if shot:
                shot = "../" + str(Path(shot).relative_to(config.ROOT))

            rows.append({
                "site_id": sid,
                "domain": m["domain"],
                "url": m["url"],
                "category": m["category_hint"] or "—",
                "cluster_id": cid,
                "style_name": style_name,
                "color_mode": lbl.get("color_mode") or "",
                "visual_density": lbl.get("visual_density") or "",
                "typography": lbl.get("typography_style") or "",
                "layout": lbl.get("layout_structure") or "",
                "motion": lbl.get("motion_intensity") or "",
                "energy": lbl.get("visual_energy") or "",
                "substyle": lbl.get("substyle_traits") or "",
                "screenshot_rel": shot,
                "x": x,
                "y": y,
            })

    return pd.DataFrame(rows) if rows else None


# ── Figure builders ────────────────────────────────────────────────────

def build_scatter(df: pd.DataFrame) -> go.Figure:
    """Main UMAP scatter colored by visual style cluster."""
    fig = go.Figure()

    noise = df[df["cluster_id"] == -1]
    clusters = df[df["cluster_id"] >= 0]

    if not noise.empty:
        fig.add_trace(go.Scattergl(
            x=noise["x"],
            y=noise["y"],
            mode="markers",
            name="Noise / Outliers",
            marker=dict(size=5, color=_NOISE_COLOR, symbol="circle"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "<span style='color:#888'>Noise — no cluster assigned</span>"
                "<extra></extra>"
            ),
            customdata=noise[["domain", "category", "screenshot_rel"]].values,
        ))

    style_order = (
        clusters.groupby("style_name")["site_id"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for i, style_name in enumerate(style_order):
        sub = clusters[clusters["style_name"] == style_name]
        color = _PALETTE[i % len(_PALETTE)]

        fig.add_trace(go.Scattergl(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=style_name,
            marker=dict(
                size=7,
                color=color,
                opacity=0.85,
                line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "Style: <i>%{customdata[2]}</i><br>"
                "Colors: %{customdata[3]}<br>"
                "Layout: %{customdata[4]}<br>"
                "Typography: %{customdata[5]}<br>"
                "<extra></extra>"
            ),
            customdata=sub[[
                "domain", "category", "style_name",
                "color_mode", "layout", "typography", "screenshot_rel",
            ]].values,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cdd8", family="Inter, system-ui, sans-serif", size=12),
        legend=dict(
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=620,
        hovermode="closest",
        dragmode="pan",
    )
    return fig


def build_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of sites per style cluster."""
    counts = (
        df[df["cluster_id"] >= 0]
        .groupby("style_name")["site_id"]
        .count()
        .sort_values(ascending=True)
    )

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(counts))]

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: <b>%{x} sites</b><extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cdd8", family="Inter, system-ui, sans-serif", size=11),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            color="#c9cdd8",
            title="",
        ),
        yaxis=dict(showgrid=False, color="#c9cdd8", automargin=True),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(300, 36 * len(counts)),
        hovermode="closest",
    )
    return fig


def build_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Category × style heatmap. Returns None if categories are too sparse."""
    subset = df[(df["cluster_id"] >= 0) & (df["category"] != "—")]
    if subset.empty or subset["category"].nunique() < 2:
        return None

    pivot = (
        subset.groupby(["category", "style_name"])["site_id"]
        .count()
        .unstack(fill_value=0)
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Purp",
        hovertemplate="%{y} × %{x}: <b>%{z} sites</b><extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#c9cdd8"),
            outlinewidth=0,
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9cdd8", family="Inter, system-ui, sans-serif", size=10),
        xaxis=dict(tickangle=-40, automargin=True, color="#c9cdd8"),
        yaxis=dict(automargin=True, color="#c9cdd8"),
        margin=dict(l=10, r=10, t=10, b=80),
        height=420,
    )
    return fig


# ── HTML assembly ──────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Website Visual Style Atlas</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        rel="stylesheet"/>
  <style>
    :root {{
      --bg:        #0a0b0f;
      --surface:   #111318;
      --surface2:  #191c25;
      --border:    rgba(255,255,255,0.08);
      --text:      #e2e4ec;
      --muted:     #6b7280;
      --accent:    #6366f1;
      --accent2:   #8b5cf6;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", system-ui, sans-serif;
      min-height: 100vh;
    }}

    /* ── Header ── */
    header {{
      padding: 2rem 3rem 1.75rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(160deg, #0d0e14 0%, #111318 100%);
    }}
    header h1 {{
      font-size: 1.9rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    header p {{
      margin-top: 0.45rem;
      color: var(--muted);
      font-size: 0.88rem;
      max-width: 780px;
      line-height: 1.6;
    }}
    .tag {{
      display: inline-block;
      margin-top: 0.8rem;
      background: rgba(99,102,241,0.15);
      border: 1px solid rgba(99,102,241,0.35);
      color: #a5b4fc;
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      margin-right: 0.4rem;
    }}

    /* ── Stats bar ── */
    .stats-bar {{
      display: flex;
      gap: 0;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    .stat {{
      flex: 1;
      padding: 1rem 1.5rem;
      border-right: 1px solid var(--border);
      text-align: center;
    }}
    .stat:last-child {{ border-right: none; }}
    .stat .val {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: -0.02em;
    }}
    .stat .lbl {{
      margin-top: 0.15rem;
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }}

    /* ── Layout ── */
    main {{
      max-width: 1600px;
      padding: 2rem 3rem;
    }}
    .section-label {{
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.75rem;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    @media (max-width: 900px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      header, .stats-bar .stat, main {{ padding-left: 1rem; padding-right: 1rem; }}
    }}

    /* ── Screenshot preview panel ── */
    #preview-panel {{
      display: none;
      position: fixed;
      right: 1.5rem;
      bottom: 1.5rem;
      width: 320px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.6);
      z-index: 9999;
      overflow: hidden;
    }}
    #preview-panel .preview-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    #preview-domain {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    #preview-close {{
      background: none;
      border: none;
      color: var(--muted);
      font-size: 1.1rem;
      cursor: pointer;
      padding: 0 0.25rem;
      line-height: 1;
    }}
    #preview-close:hover {{ color: var(--text); }}
    #preview-img {{
      width: 100%;
      display: block;
      aspect-ratio: 16/10;
      object-fit: cover;
      background: var(--surface);
    }}
    #preview-meta {{
      padding: 0.6rem 1rem;
      font-size: 0.75rem;
      color: var(--muted);
      line-height: 1.6;
    }}

    /* ── Footer ── */
    footer {{
      padding: 1.5rem 3rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.78rem;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
  </style>
</head>
<body>

<header>
  <h1>Website Visual Style Atlas</h1>
  <p>
    UMAP projection of visual embeddings across <strong>{n_sites}</strong> websites,
    clustered into <strong>{n_clusters}</strong> distinct style groups using
    CLIP&nbsp;ViT-B/32&nbsp;+&nbsp;HDBSCAN, with clusters labelled by Claude&nbsp;VLM.
  </p>
  <div>
    <span class="tag">CLIP</span>
    <span class="tag">UMAP</span>
    <span class="tag">HDBSCAN</span>
    <span class="tag">Claude VLM</span>
    <span class="tag">Plotly</span>
  </div>
</header>

<div class="stats-bar">
  <div class="stat">
    <div class="val">{n_sites}</div>
    <div class="lbl">Websites</div>
  </div>
  <div class="stat">
    <div class="val">{n_clusters}</div>
    <div class="lbl">Style Clusters</div>
  </div>
  <div class="stat">
    <div class="val">{n_categories}</div>
    <div class="lbl">Categories</div>
  </div>
  <div class="stat">
    <div class="val">{noise_pct}%</div>
    <div class="lbl">Noise</div>
  </div>
  <div class="stat">
    <div class="val">{run_id}</div>
    <div class="lbl">Run ID</div>
  </div>
</div>

<main>
  <div class="card">
    <div class="section-label">Visual Style Map — hover to explore · click for screenshot preview</div>
    {scatter_div}
  </div>

  <div class="two-col">
    <div class="card">
      <div class="section-label">Sites per Style Cluster</div>
      {bar_div}
    </div>
    <div class="card">
      <div class="section-label">Category × Style Distribution</div>
      {heatmap_div}
    </div>
  </div>
</main>

<!-- Screenshot preview panel -->
<div id="preview-panel">
  <div class="preview-header">
    <span id="preview-domain">—</span>
    <button id="preview-close">✕</button>
  </div>
  <img id="preview-img" src="" alt="Screenshot"/>
  <div id="preview-meta"></div>
</div>

{plotlyjs}

<script>
(function () {{
  var panel   = document.getElementById("preview-panel");
  var pDomain = document.getElementById("preview-domain");
  var pImg    = document.getElementById("preview-img");
  var pMeta   = document.getElementById("preview-meta");
  var pClose  = document.getElementById("preview-close");

  pClose.onclick = function () {{ panel.style.display = "none"; }};

  // Attach to the scatter figure after Plotly renders it
  var scatterEl = document.getElementById("scatter-fig");
  if (scatterEl) {{
    scatterEl.on("plotly_click", function (data) {{
      var pt = data.points[0];
      if (!pt || !pt.customdata) return;
      var cd = pt.customdata;
      // cd: [domain, category, style_name, color_mode, layout, typography, screenshot_rel]
      var domain  = cd[0] || "";
      var cat     = cd[1] || "";
      var style   = cd[2] || "";
      var color   = cd[3] || "";
      var layout  = cd[4] || "";
      var typo    = cd[5] || "";
      var imgPath = cd[6] || "";

      pDomain.textContent = domain;
      pMeta.innerHTML = [
        cat     ? "<b>Category:</b> " + cat     : "",
        style   ? "<b>Style:</b> "   + style   : "",
        color   ? "<b>Color:</b> "   + color   : "",
        layout  ? "<b>Layout:</b> "  + layout  : "",
        typo    ? "<b>Type:</b> "    + typo    : "",
      ].filter(Boolean).join("<br/>");

      if (imgPath) {{
        pImg.src    = imgPath;
        pImg.style.display = "block";
      }} else {{
        pImg.src    = "";
        pImg.style.display = "none";
      }}
      panel.style.display = "block";
    }});
  }}
}})();
</script>

<footer>
  <span>Generated {date} · Run ID: {run_id}</span>
  <span>Pipeline: CLIP ViT-B/32 → UMAP (20D cluster / 2D viz) → HDBSCAN → Claude VLM labels</span>
</footer>
</body>
</html>
"""


def _fig_to_div(fig: go.Figure, div_id: str, include_js: bool) -> str:
    """Render a Plotly figure to an HTML div string."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_js,
        div_id=div_id,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )


def render_html(df: pd.DataFrame, run_id: str, output_path: Path) -> None:
    n_sites = len(df)
    n_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique()
    n_categories = df[df["category"] != "—"]["category"].nunique()
    n_noise = (df["cluster_id"] == -1).sum()
    noise_pct = round(100 * n_noise / n_sites) if n_sites else 0

    console.print(f"[bold]Building scatter ({n_sites} points)...[/]")
    scatter_fig = build_scatter(df)
    scatter_div = _fig_to_div(scatter_fig, "scatter-fig", include_js=False)

    console.print("[bold]Building cluster size bar chart...[/]")
    bar_fig = build_bar(df)
    bar_div = _fig_to_div(bar_fig, "bar-fig", include_js=False)

    console.print("[bold]Building category heatmap...[/]")
    heatmap_fig = build_heatmap(df)
    if heatmap_fig:
        heatmap_div = _fig_to_div(heatmap_fig, "heatmap-fig", include_js=False)
    else:
        heatmap_div = (
            "<p style='color:#6b7280; padding:2rem; text-align:center; font-size:0.85rem;'>"
            "Not enough category variety to show a heatmap.<br/>"
            "Add <code>category_hint</code> values to <code>seeds.csv</code>.</p>"
        )

    # Embed Plotly.js once, as a CDN script tag (keeps file small)
    plotlyjs_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    html = _HTML_TEMPLATE.format(
        n_sites=n_sites,
        n_clusters=n_clusters,
        n_categories=n_categories,
        noise_pct=noise_pct,
        run_id=run_id,
        scatter_div=scatter_div,
        bar_div=bar_div,
        heatmap_div=heatmap_div,
        plotlyjs=plotlyjs_tag,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML atlas of visual website styles"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Clustering run ID (default: latest)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUTS_DIR / "website_atlas.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering run found. Run 03_cluster.py first.[/]")
        conn.close()
        return

    console.print(f"[bold]Visualizing run:[/] {run_id}")

    df = build_dataframe(conn, run_id)
    conn.close()

    if df is None or df.empty:
        console.print("[red]No data to visualize.[/]")
        return

    output_path = Path(args.output)
    render_html(df, run_id, output_path)

    size_kb = round(output_path.stat().st_size / 1024)
    console.print(f"\n[bold green]Atlas exported → {output_path}[/] ({size_kb} KB)")
    console.print(
        f"  [dim]{len(df)} sites · "
        f"{df[df['cluster_id'] >= 0]['cluster_id'].nunique()} clusters · "
        f"{(df['cluster_id'] == -1).sum()} noise points[/]"
    )
    console.print(f"\n  Open in browser: [cyan]open {output_path}[/]")


if __name__ == "__main__":
    main()
