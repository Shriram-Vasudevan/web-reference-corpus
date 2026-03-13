#!/usr/bin/env python3
"""Step 6: Build industry × style profiles from labeled catalog.

Aggregates (industry, visual_style, color_mode, business_model, brand_tier) combinations
across all labeled clusters to produce a set of named archetypes. Each archetype captures
a recurring pattern like "fintech dark minimal" or "health light clean" with representative
domains and a generated description.

The output — industry_style_profiles.json — is consumed by Superpowering-Agents retrieval
to enable combined industry+style scoring during reference lookup.

Usage:
    ./.venv/bin/python scripts/06_build_industry_profiles.py
    ./.venv/bin/python scripts/06_build_industry_profiles.py --min-clusters 2
    ./.venv/bin/python scripts/06_build_industry_profiles.py --output-dir /path/to/Superpowering-Agents/reference_data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.storage import get_connection, init_db, get_latest_run_id, get_all_style_labels, get_cluster_members

console = Console()

# Canonical industry groups — broader categories that contain multiple specific industries.
# Used to compute "related industry" partial-match scores in the retrieval layer.
INDUSTRY_GROUPS = {
    "finance_group": ["finance", "fintech", "insurance", "banking", "crypto_web3", "defi"],
    "health_group": ["health", "fitness_wellness", "mental_health"],
    "tech_group": ["technology", "saas", "ai_ml", "developer_tools", "devops", "security"],
    "ecommerce_group": ["ecommerce", "fashion", "beauty", "food_beverage", "restaurant"],
    "media_group": ["media", "news", "podcast", "entertainment", "sports"],
    "education_group": ["education", "edtech"],
    "creative_group": ["creative_agency", "design_studio"],
    "real_estate_group": ["real_estate", "proptech"],
    "travel_group": ["travel", "hospitality"],
    "gaming_group": ["gaming", "esports"],
}

# Well-known archetypes with their expected industry+style+color signatures.
# These seed the profile keys so they're stable and human-readable.
KNOWN_ARCHETYPES: list[dict] = [
    {
        "key": "fintech_dark_minimal",
        "industries": ["fintech", "banking", "defi", "crypto_web3"],
        "visual_styles": ["minimal dark", "dark gradient", "glassmorphism", "dark minimal"],
        "color_modes": ["dark"],
        "description": "Premium dark UI, high trust signal, data-forward. Stripe, Linear aesthetic.",
    },
    {
        "key": "fintech_light_clean",
        "industries": ["fintech", "banking", "insurance", "finance"],
        "visual_styles": ["minimal light", "corporate clean", "clean light"],
        "color_modes": ["light"],
        "description": "Light, trustworthy fintech. Approachable but professional.",
    },
    {
        "key": "health_light_clean",
        "industries": ["health", "fitness_wellness", "mental_health"],
        "visual_styles": ["minimal light", "soft pastel", "clean clinical", "light clean"],
        "color_modes": ["light"],
        "description": "Approachable health/wellness. White space dominant, clinical trust.",
    },
    {
        "key": "saas_dark_modern",
        "industries": ["saas", "ai_ml", "technology"],
        "visual_styles": ["dark gradient", "glassmorphism", "minimal dark", "dark modern"],
        "color_modes": ["dark"],
        "description": "Modern SaaS dark theme. Often gradient hero, feature cards, clean nav.",
    },
    {
        "key": "saas_light_minimal",
        "industries": ["saas", "ai_ml", "technology", "developer_tools"],
        "visual_styles": ["minimal light", "clean light", "light clean", "corporate clean"],
        "color_modes": ["light"],
        "description": "Clean minimal SaaS. Lots of white space, product screenshots, clear CTAs.",
    },
    {
        "key": "developer_tools_mono",
        "industries": ["developer_tools", "devops", "security", "open_source"],
        "visual_styles": ["monospace dark", "minimal dark", "dark technical", "terminal style"],
        "color_modes": ["dark"],
        "description": "Developer-first tooling. Terminal/code aesthetics, dark mode, functional UI.",
    },
    {
        "key": "ecommerce_bold_colorful",
        "industries": ["ecommerce", "fashion", "beauty"],
        "visual_styles": ["vibrant illustrated", "bold colorful", "colorful modern"],
        "color_modes": ["colorful", "mixed"],
        "description": "Bold consumer ecommerce. High-contrast products, vibrant photography.",
    },
    {
        "key": "ecommerce_minimal_premium",
        "industries": ["ecommerce", "fashion", "beauty", "luxury_premium"],
        "visual_styles": ["minimal light", "luxury minimal", "editorial clean"],
        "color_modes": ["light", "monochrome"],
        "description": "Premium/luxury e-commerce. Restrained palette, large product imagery.",
    },
    {
        "key": "creative_agency_bold",
        "industries": ["creative_agency", "design_studio"],
        "visual_styles": ["neo-brutalist", "bold typographic", "creative experimental", "bold modern"],
        "color_modes": ["colorful", "mixed", "dark"],
        "description": "Agency portfolio. Experimental layouts, strong typography, creative risk-taking.",
    },
    {
        "key": "edtech_light_approachable",
        "industries": ["education", "edtech"],
        "visual_styles": ["minimal light", "colorful friendly", "vibrant illustrated"],
        "color_modes": ["light", "colorful"],
        "description": "Educational platform. Friendly, approachable, often illustrated.",
    },
    {
        "key": "media_editorial",
        "industries": ["media", "news", "podcast"],
        "visual_styles": ["serif editorial", "editorial clean", "editorial scroll"],
        "color_modes": ["light", "dark"],
        "description": "Content-first editorial. Strong typography, article-centric layout.",
    },
    {
        "key": "gaming_dark_immersive",
        "industries": ["gaming", "esports"],
        "visual_styles": ["dark gradient", "immersive dark", "futuristic dark"],
        "color_modes": ["dark"],
        "description": "Gaming/esports dark UI. Immersive, action-forward, dramatic visuals.",
    },
    {
        "key": "travel_rich_visual",
        "industries": ["travel", "hospitality", "real_estate"],
        "visual_styles": ["fullscreen media", "rich visual", "photography forward"],
        "color_modes": ["mixed", "light"],
        "description": "Visual-first travel/hospitality. Full-bleed photography, hero-driven.",
    },
    {
        "key": "startup_modern_gradient",
        "industries": ["saas", "ai_ml", "technology", "fintech"],
        "visual_styles": ["gradient modern", "colorful gradient", "vibrant gradient"],
        "color_modes": ["colorful", "mixed"],
        "description": "Modern startup with gradient branding. Bold color, energetic, growth-stage feel.",
    },
    {
        "key": "enterprise_corporate",
        "industries": ["technology", "saas", "finance", "insurance", "logistics", "hr_recruiting"],
        "visual_styles": ["corporate clean", "classic corporate", "minimal corporate"],
        "color_modes": ["light"],
        "description": "Enterprise B2B. Professional, trustworthy, conservative layout.",
    },
    {
        "key": "nonprofit_warm",
        "industries": ["nonprofit", "health", "education"],
        "visual_styles": ["warm minimal", "soft colorful", "illustrated friendly"],
        "color_modes": ["light", "colorful"],
        "description": "Mission-driven nonprofit. Warm, human-centered, community-focused.",
    },
    {
        "key": "food_vibrant",
        "industries": ["food_beverage", "restaurant"],
        "visual_styles": ["rich photography", "warm editorial", "vibrant illustrated"],
        "color_modes": ["colorful", "mixed"],
        "description": "Food & beverage. Appetite-inducing photography, warm tones, editorial layout.",
    },
]


def _normalize_style(s: str) -> str:
    return s.lower().strip()


def _style_matches_archetype(visual_style: str, archetype_styles: list[str]) -> bool:
    """Fuzzy match: true if any word from the archetype style appears in the visual style."""
    vs = _normalize_style(visual_style)
    for arch_style in archetype_styles:
        arch_words = set(_normalize_style(arch_style).split())
        vs_words = set(vs.split())
        if arch_words & vs_words:
            return True
    return False


def _assign_archetype(industry: str, visual_style: str, color_mode: str) -> str | None:
    """Find the best matching archetype key for a labeled cluster."""
    candidates = []
    for arch in KNOWN_ARCHETYPES:
        industry_match = industry in arch["industries"]
        color_match = color_mode in arch["color_modes"] if arch["color_modes"] else True
        style_match = _style_matches_archetype(visual_style, arch["visual_styles"])

        if industry_match and (color_match or style_match):
            score = int(industry_match) + int(color_match) + int(style_match)
            candidates.append((score, arch["key"]))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_profiles(
    conn,
    run_id: str,
    min_clusters: int = 1,
) -> dict:
    """Build the full industry_style_profiles structure from labeled data."""
    labels = get_all_style_labels(conn, run_id)

    # Map archetype key → list of cluster data
    archetype_clusters: dict[str, list[dict]] = defaultdict(list)
    unmatched = 0

    for label in labels:
        label_dict = dict(label)
        industry = label_dict.get("industry") or "general"
        visual_style = label_dict.get("visual_style") or ""
        color_mode = label_dict.get("color_mode") or "mixed"
        quality_score = label_dict.get("quality_score") or 0
        cluster_id = label_dict["cluster_id"]

        if not industry or industry in ("non_applicable", "general") or quality_score < 2:
            continue

        archetype_key = _assign_archetype(industry, visual_style, color_mode)
        if archetype_key:
            archetype_clusters[archetype_key].append({
                "cluster_id": cluster_id,
                "industry": industry,
                "visual_style": visual_style,
                "color_mode": color_mode,
                "quality_score": quality_score,
                "business_model": label_dict.get("business_model") or "unknown",
                "brand_tier": label_dict.get("brand_tier") or "unknown",
            })
        else:
            unmatched += 1

    # Assign profile keys back to DB
    for label in labels:
        label_dict = dict(label)
        industry = label_dict.get("industry") or "general"
        visual_style = label_dict.get("visual_style") or ""
        color_mode = label_dict.get("color_mode") or "mixed"
        key = _assign_archetype(industry, visual_style, color_mode)
        if key:
            from src.utils.storage import update_industry_style_profile
            update_industry_style_profile(conn, label_dict["cluster_id"], run_id, key)

    # Collect representative domains per archetype
    profiles = {}
    for arch in KNOWN_ARCHETYPES:
        key = arch["key"]
        clusters = archetype_clusters.get(key, [])

        if len(clusters) < min_clusters:
            # Still include it — may be populated after next labeling run
            profiles[key] = {
                **arch,
                "cluster_count": 0,
                "example_domains": [],
                "avg_quality": None,
                "observed_visual_styles": [],
                "observed_color_modes": [],
                "observed_business_models": [],
                "observed_brand_tiers": [],
            }
            continue

        # Collect members for representative domains
        all_domains = []
        for c in clusters:
            members = get_cluster_members(conn, run_id, c["cluster_id"])
            for m in members:
                if m["domain"]:
                    all_domains.append(m["domain"])

        avg_q = sum(c["quality_score"] for c in clusters) / len(clusters)

        profiles[key] = {
            **arch,
            "cluster_count": len(clusters),
            "example_domains": list(dict.fromkeys(all_domains))[:10],  # dedup, keep order
            "avg_quality": round(avg_q, 2),
            "observed_visual_styles": list({c["visual_style"] for c in clusters if c["visual_style"]}),
            "observed_color_modes": list({c["color_mode"] for c in clusters if c["color_mode"]}),
            "observed_business_models": list({c["business_model"] for c in clusters if c["business_model"] != "unknown"}),
            "observed_brand_tiers": list({c["brand_tier"] for c in clusters if c["brand_tier"] != "unknown"}),
        }

    return {
        "version": "1.0",
        "run_id": run_id,
        "industry_groups": INDUSTRY_GROUPS,
        "profiles": profiles,
        "total_profiles": len(profiles),
        "populated_profiles": sum(1 for p in profiles.values() if p["cluster_count"] > 0),
        "unmatched_clusters": unmatched,
    }


def main():
    parser = argparse.ArgumentParser(description="Build industry × style profiles")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--min-clusters", type=int, default=1,
                        help="Min clusters needed to mark a profile as populated")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR),
                        help="Directory to write industry_style_profiles.json")
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)

    run_id = args.run_id or get_latest_run_id(conn)
    if not run_id:
        console.print("[red]No clustering runs found.[/]")
        conn.close()
        return

    console.print(f"[bold]Using run_id:[/] {run_id}")
    console.print("Building industry × style profiles...")

    profiles_data = build_profiles(conn, run_id, min_clusters=args.min_clusters)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "industry_style_profiles.json"

    with open(output_path, "w") as f:
        json.dump(profiles_data, f, indent=2)

    console.print(f"\n[bold green]Wrote {output_path}[/]")
    console.print(f"  Profiles: {profiles_data['total_profiles']}")
    console.print(f"  Populated (≥{args.min_clusters} cluster): {profiles_data['populated_profiles']}")
    console.print(f"  Unmatched clusters: {profiles_data['unmatched_clusters']}")

    # Summary table
    table = Table(title="Profile Summary")
    table.add_column("Profile Key")
    table.add_column("Clusters", justify="right")
    table.add_column("Avg Quality", justify="right")
    table.add_column("Example Domains")

    for key, profile in sorted(profiles_data["profiles"].items(), key=lambda x: -x[1]["cluster_count"]):
        domains_preview = ", ".join(profile["example_domains"][:3]) or "—"
        avg_q = f"{profile['avg_quality']:.1f}" if profile["avg_quality"] else "—"
        table.add_row(key, str(profile["cluster_count"]), avg_q, domains_preview)

    console.print()
    console.print(table)

    conn.close()


if __name__ == "__main__":
    main()
