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
  "industry": "The specific industry or vertical. Pick ONE from: technology | saas | ai_ml | health | fitness_wellness | mental_health | finance | fintech | insurance | banking | ecommerce | fashion | beauty | education | edtech | creative_agency | design_studio | media | news | podcast | food_beverage | restaurant | real_estate | proptech | travel | hospitality | gaming | esports | crypto_web3 | defi | developer_tools | devops | security | logistics | hr_recruiting | legal | nonprofit | sports | entertainment | automotive | general | non_applicable",
  "industry_confidence": "Float 0.0-1.0. How confident are you in the industry label? 1.0 = unmistakable (e.g. a hospital site), 0.7 = fairly clear, 0.5 = ambiguous or generic, 0.3 = guessing",
  "business_model": "How the business operates. Pick ONE from: b2b_saas | b2c_consumer | marketplace | developer_tool | agency_studio | enterprise_software | open_source | media_content | ecommerce_store | nonprofit | platform | api_service | unknown",
  "brand_tier": "The brand positioning archetype. Pick ONE from: startup_modern | enterprise_trusted | minimal_premium | consumer_playful | developer_focused | creative_bold | corporate_formal | luxury_premium | community_driven | unknown",
  "color_mode": "one of: light | dark | mixed | colorful | monochrome",
  "layout_pattern": "one of: hero_cta | split_screen | card_grid | single_column | dashboard_panels | form_centered | editorial_scroll | fullscreen_media | sidebar_content | multi_section_scroll",
  "typography_style": "one of: sans_clean | serif_editorial | display_bold | monospace_technical | mixed_expressive",
  "design_era": "one of: modern_2024 | classic_corporate | retro_vintage | futuristic | timeless_minimal",
  "target_audience": "one of: developers | consumers | enterprise | designers | general_public | investors",
  "distinguishing_features": ["list", "of", "2-4", "specific", "features", "that make this cluster unique and worth referencing"]
}}

Industry classification guidance:
- Prefer specific sub-industries over broad ones: "fintech" > "finance", "edtech" > "education", "fitness_wellness" > "health"
- "technology" = generic tech company with no clear vertical; use a specific industry if possible
- "developer_tools" = products built FOR developers (IDEs, CLIs, APIs, SDKs)
- "devops" = CI/CD, monitoring, infrastructure, cloud platforms
- "saas" = B2B software product that doesn't fit a more specific vertical
- "general" = genuinely impossible to classify (portfolio, personal site, etc.)

Focus on what makes these sites USEFUL as design references. A developer asking "show me great SaaS pricing pages" should find them through your page_type + industry labels. Be honest with quality_score — not everything is a 5."""


INDUSTRY_RECLASSIFY_SYSTEM = """You are an expert business analyst classifying websites by industry and business model.
You will see screenshots of websites and must identify exactly what industry they serve and how the business operates.
Your output will be used to match design references to user projects by industry — precision matters more than speed."""

INDUSTRY_RECLASSIFY_USER = """These {count} website screenshots belong to the same cluster, currently labeled as industry="{current_industry}" (confidence={current_confidence}).

Re-examine them with fresh eyes and return ONLY a JSON object:

{{
  "industry": "Pick ONE from the full taxonomy: technology | saas | ai_ml | health | fitness_wellness | mental_health | finance | fintech | insurance | banking | ecommerce | fashion | beauty | education | edtech | creative_agency | design_studio | media | news | podcast | food_beverage | restaurant | real_estate | proptech | travel | hospitality | gaming | esports | crypto_web3 | defi | developer_tools | devops | security | logistics | hr_recruiting | legal | nonprofit | sports | entertainment | automotive | general | non_applicable",
  "industry_confidence": "Float 0.0-1.0 — how certain are you?",
  "business_model": "Pick ONE: b2b_saas | b2c_consumer | marketplace | developer_tool | agency_studio | enterprise_software | open_source | media_content | ecommerce_store | nonprofit | platform | api_service | unknown",
  "brand_tier": "Pick ONE: startup_modern | enterprise_trusted | minimal_premium | consumer_playful | developer_focused | creative_bold | corporate_formal | luxury_premium | community_driven | unknown",
  "reasoning": "One sentence: what visual/contextual signals led to this classification?"
}}

Prefer specific sub-industries over broad ones. If the previous label looks correct, you may keep it — but reassign if you see clearer evidence."""


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
