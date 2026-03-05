"""Prompt templates for Claude VLM style labeling."""

CLUSTER_LABEL_SYSTEM = """You are an expert UI/UX analyst classifying websites for a design reference system.
Your classifications will be used by an LLM to retrieve the best reference designs when someone says
"I'm building a [type] for [industry]" — so your labels must be functional and actionable, not just aesthetic.

You will be shown screenshots from websites that have been grouped together by visual similarity.
Your job is to identify WHAT kind of page this is and HOW it looks, so the right references get retrieved."""

CLUSTER_LABEL_USER = """These {count} website screenshots belong to the same visual cluster.

Analyze them and respond with ONLY a JSON object (no markdown, no explanation):

{{
  "page_type": "The functional type of page. Pick ONE from: landing_page | product_page | pricing | dashboard | auth_login | blog | portfolio | ecommerce | documentation | error_page | agency_showcase | saas_app | marketing_site | corporate | coming_soon | creative_experimental | directory_listing | news_media",
  "visual_style": "A concise 2-3 word aesthetic label (e.g., 'minimal light', 'dark gradient', 'bold typographic', 'glassmorphism', 'neo-brutalist', 'corporate clean', 'vibrant illustrated')",
  "quality_score": "Rate the UI design quality from 1-5. 5 = best-in-class design worth copying. 4 = very good, professional. 3 = competent, nothing special. 2 = below average. 1 = broken, error page, or non-functional.",
  "industry": "The industry or vertical. Pick ONE from: technology | saas | ai_ml | health | finance | ecommerce | education | creative_agency | media | food_beverage | real_estate | travel | gaming | crypto_web3 | developer_tools | general | non_applicable",
  "color_mode": "one of: light | dark | mixed | colorful | monochrome",
  "layout_pattern": "one of: hero_cta | split_screen | card_grid | single_column | dashboard_panels | form_centered | editorial_scroll | fullscreen_media | sidebar_content | multi_section_scroll",
  "typography_style": "one of: sans_clean | serif_editorial | display_bold | monospace_technical | mixed_expressive",
  "design_era": "one of: modern_2024 | classic_corporate | retro_vintage | futuristic | timeless_minimal",
  "target_audience": "one of: developers | consumers | enterprise | designers | general_public | investors",
  "distinguishing_features": ["list", "of", "2-4", "specific", "features", "that make this cluster unique and worth referencing"]
}}

Focus on what makes these sites USEFUL as design references. A developer asking "show me great SaaS pricing pages" should find them through your page_type + industry labels. Be honest with quality_score — not everything is a 5."""

STYLE_CATALOG_PROMPT = """You are helping build a design reference catalog.
Given this classification, write a 1-2 sentence description that helps someone decide
if these reference designs match what they're trying to build.

Page type: {page_type}
Visual style: {visual_style}
Quality: {quality_score}/5
Industry: {industry}
Color: {color_mode}, Layout: {layout_pattern}, Typography: {typography_style}
Features: {distinguishing_features}

Respond with ONLY the description, no formatting."""
