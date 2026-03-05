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

# Muted, sophisticated palette for white backgrounds
_PALETTE = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483", "#7b2d8e",
    "#2d6a4f", "#40916c", "#1b4332", "#264653", "#2a9d8f",
    "#6c584c", "#a98467", "#774936", "#3d405b", "#5f0f40",
    "#9a031e", "#0b525b", "#065a60", "#3a0ca3", "#4361ee",
    "#560bad", "#7209b7", "#b5179e", "#480ca8", "#023e8a",
]
_NOISE_COLOR = "rgba(180, 180, 190, 0.45)"


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
        page_type = lbl.get("page_type") or ""
        visual_style = lbl.get("visual_style") or ""
        quality = lbl.get("quality_score") or 0

        if cid == -1:
            style_name = "Noise / Outliers"
        elif page_type and visual_style:
            style_name = f"{page_type} — {visual_style}"
        else:
            style_name = f"Cluster {cid}"

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
                "page_type": page_type,
                "visual_style": visual_style,
                "quality_score": quality,
                "industry": lbl.get("industry") or "",
                "color_mode": lbl.get("color_mode") or "",
                "layout_pattern": lbl.get("layout_pattern") or "",
                "typography": lbl.get("typography_style") or "",
                "design_era": lbl.get("design_era") or "",
                "target_audience": lbl.get("target_audience") or "",
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
            marker=dict(size=4, color=_NOISE_COLOR, symbol="circle"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "<span style='color:#999'>Noise — no cluster assigned</span>"
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
                opacity=0.8,
                line=dict(width=0.5, color="rgba(255,255,255,0.9)"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Page: %{customdata[1]} · Style: %{customdata[2]}<br>"
                "Quality: %{customdata[3]}/5 · Industry: %{customdata[4]}<br>"
                "Layout: %{customdata[5]} · Colors: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
            customdata=sub[[
                "domain", "page_type", "visual_style",
                "quality_score", "industry", "layout_pattern",
                "color_mode", "screenshot_rel",
            ]].values,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a2e", family="Inter, system-ui, sans-serif", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0.06)",
            borderwidth=1,
            font=dict(size=11, color="#3d405b"),
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
        font=dict(color="#3d405b", family="Inter, system-ui, sans-serif", size=11),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            color="#3d405b",
            title="",
        ),
        yaxis=dict(showgrid=False, color="#3d405b", automargin=True),
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
        colorscale=[[0, "#ffffff"], [0.25, "#e8e0f0"], [0.5, "#b8a9d4"], [0.75, "#6b4fa0"], [1, "#1a1a2e"]],
        hovertemplate="%{y} × %{x}: <b>%{z} sites</b><extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#3d405b"),
            outlinewidth=0,
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3d405b", family="Inter, system-ui, sans-serif", size=10),
        xaxis=dict(tickangle=-40, automargin=True, color="#3d405b"),
        yaxis=dict(automargin=True, color="#3d405b"),
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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
        rel="stylesheet"/>
  {plotlyjs}
  <style>
    :root {{
      --bg:        #ffffff;
      --surface:   #fafafa;
      --surface2:  #f5f5f5;
      --border:    rgba(0,0,0,0.08);
      --border2:   rgba(0,0,0,0.04);
      --text:      #1a1a2e;
      --text-sec:  #3d405b;
      --muted:     #8d8d92;
      --accent:    #1a1a2e;
      --accent2:   #3d405b;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }}

    /* ── Header ── */
    header {{
      padding: 4rem 4rem 3rem;
      max-width: 1400px;
      margin: 0 auto;
    }}
    header h1 {{
      font-size: 3rem;
      font-weight: 700;
      letter-spacing: -0.035em;
      color: var(--text);
      line-height: 1.1;
      max-width: 900px;
    }}
    header h1 span {{
      color: var(--muted);
    }}
    header p {{
      margin-top: 1.25rem;
      color: var(--muted);
      font-size: 0.95rem;
      max-width: 680px;
      line-height: 1.7;
      font-weight: 400;
    }}
    .tag-row {{
      display: flex;
      gap: 0.5rem;
      margin-top: 1.5rem;
      flex-wrap: wrap;
    }}
    .tag {{
      display: inline-block;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text-sec);
      font-size: 0.68rem;
      font-weight: 500;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 0.3rem 0.7rem;
      border-radius: 4px;
    }}

    /* ── Divider ── */
    .divider {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 0 4rem;
      max-width: 1400px;
    }}

    /* ── Stats bar ── */
    .stats-bar {{
      display: flex;
      gap: 0;
      max-width: 1400px;
      margin: 0 auto;
      padding: 0 4rem;
    }}
    .stat {{
      flex: 1;
      padding: 1.75rem 0;
      text-align: left;
    }}
    .stat .val {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.03em;
    }}
    .stat .lbl {{
      margin-top: 0.2rem;
      font-size: 0.72rem;
      font-weight: 500;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    /* ── Layout ── */
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 2.5rem 4rem 3rem;
    }}
    .section-label {{
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 1rem;
    }}
    .card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
      transition: box-shadow 0.2s ease;
    }}
    .card:hover {{
      box-shadow: 0 4px 24px rgba(0,0,0,0.04);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2rem;
      margin-bottom: 2rem;
    }}
    @media (max-width: 900px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 1.5rem; padding-right: 1.5rem; }}
      .stats-bar {{ padding: 0 1.5rem; }}
      .divider {{ margin: 0 1.5rem; }}
      header h1 {{ font-size: 2rem; }}
    }}

    /* ── Screenshot preview panel ── */
    #preview-panel {{
      display: none;
      position: fixed;
      right: 2rem;
      bottom: 2rem;
      width: 340px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.12), 0 4px 16px rgba(0,0,0,0.06);
      z-index: 9999;
      overflow: hidden;
    }}
    #preview-panel .preview-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.85rem 1.1rem;
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
      padding: 0.2rem 0.35rem;
      line-height: 1;
      border-radius: 6px;
      transition: background 0.15s;
    }}
    #preview-close:hover {{
      background: var(--surface2);
      color: var(--text);
    }}
    #preview-img {{
      width: 100%;
      display: block;
      aspect-ratio: 16/10;
      object-fit: cover;
      background: var(--surface);
    }}
    #preview-meta {{
      padding: 0.75rem 1.1rem;
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.7;
    }}
    #preview-meta b {{
      color: var(--text-sec);
      font-weight: 500;
    }}

    /* ── Footer ── */
    footer {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 2rem 4rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.78rem;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}

    /* ── Plotly overrides ── */
    .js-plotly-plot .plotly .modebar {{
      right: 8px !important;
    }}
    .js-plotly-plot .plotly .modebar-btn path {{
      fill: var(--muted) !important;
    }}
    .js-plotly-plot .plotly .modebar-btn:hover path {{
      fill: var(--text) !important;
    }}
  </style>
</head>
<body>

<header>
  <h1>Visual Style Atlas across <span>{n_sites}</span> websites, clustered into <span>{n_clusters}</span> distinct style groups.</h1>
  <p>
    UMAP projection of CLIP ViT-B/32 visual embeddings, clustered with HDBSCAN,
    and labelled by Claude VLM. Hover to explore. Click for screenshot preview.
  </p>
  <div class="tag-row">
    <span class="tag">CLIP</span>
    <span class="tag">UMAP</span>
    <span class="tag">HDBSCAN</span>
    <span class="tag">Claude VLM</span>
    <span class="tag">Plotly</span>
  </div>
</header>

<hr class="divider"/>

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
    <div class="val" style="font-size:1rem; padding-top:0.45rem;">{run_id}</div>
    <div class="lbl">Run ID</div>
  </div>
</div>

<hr class="divider"/>

<main>
  <div class="card">
    <div class="section-label">Visual Style Map</div>
    {scatter_div}
  </div>

  <div class="two-col">
    <div class="card">
      <div class="section-label">Sites per Style Cluster</div>
      {bar_div}
    </div>
    <div class="card">
      <div class="section-label">Category &times; Style Distribution</div>
      {heatmap_div}
    </div>
  </div>
</main>

<!-- Screenshot preview panel -->
<div id="preview-panel">
  <div class="preview-header">
    <span id="preview-domain">&mdash;</span>
    <button id="preview-close">&times;</button>
  </div>
  <img id="preview-img" src="" alt="Screenshot"/>
  <div id="preview-meta"></div>
</div>

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
      var domain   = cd[0] || "";
      var pageType = cd[1] || "";
      var vStyle   = cd[2] || "";
      var quality  = cd[3] || "";
      var industry = cd[4] || "";
      var layout   = cd[5] || "";
      var color    = cd[6] || "";
      var imgPath  = cd[7] || "";

      pDomain.textContent = domain;
      pMeta.innerHTML = [
        pageType ? "<b>Page:</b> " + pageType    : "",
        vStyle   ? "<b>Style:</b> " + vStyle      : "",
        quality  ? "<b>Quality:</b> " + quality + "/5" : "",
        industry ? "<b>Industry:</b> " + industry : "",
        layout   ? "<b>Layout:</b> " + layout     : "",
        color    ? "<b>Color:</b> " + color       : "",
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
  <span>Generated {date} &middot; Run ID: {run_id}</span>
  <span>CLIP ViT-B/32 &rarr; UMAP &rarr; HDBSCAN &rarr; Claude VLM</span>
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
            "<p style='color:#8d8d92; padding:2rem; text-align:center; font-size:0.85rem;'>"
            "Not enough category variety to show a heatmap.<br/>"
            "Add <code>category_hint</code> values to <code>data/sources/website_urls.csv</code>.</p>"
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
