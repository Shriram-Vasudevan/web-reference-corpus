"""Prompt templates for Claude VLM style labeling."""

CLUSTER_LABEL_SYSTEM = """You are a visual design expert who classifies website styles.
You will be shown screenshots from websites that have been grouped together by visual similarity.
Analyze the shared visual style and return a structured JSON description."""

CLUSTER_LABEL_USER = """These {count} website screenshots belong to the same visual style cluster.

Analyze their shared visual characteristics and respond with ONLY a JSON object (no markdown, no explanation) with these fields:

{{
  "umbrella_label": "A concise 2-4 word style label (e.g., 'Dark Minimal SaaS', 'Vibrant Editorial Grid', 'Clean Corporate White')",
  "substyle_traits": ["list", "of", "3-5", "specific", "visual", "traits"],
  "visual_density": "one of: sparse | moderate | dense | very_dense",
  "color_mode": "one of: light | dark | mixed | colorful | monochrome",
  "typography_style": "one of: sans_serif_clean | serif_editorial | mixed_expressive | monospace_technical | display_bold",
  "layout_structure": "one of: single_column | two_column | grid | asymmetric | hero_focused | dashboard | card_based",
  "motion_intensity": "one of: static | subtle | moderate | heavy",
  "visual_energy": "one of: calm | balanced | energetic | intense"
}}

Be precise and specific. The label should capture what makes this cluster visually distinct."""

STYLE_CATALOG_PROMPT = """You are helping build a style catalog for website design.
Given this style descriptor, write a 1-2 sentence natural language description
that a web designer or LLM could use to understand and reproduce this style.

Style: {label}
Traits: {traits}
Density: {density}, Color: {color}, Typography: {typography}
Layout: {layout}, Motion: {motion}, Energy: {energy}

Respond with ONLY the description, no formatting."""
