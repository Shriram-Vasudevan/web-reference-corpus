"""
Scrape curated website lists from multiple sources to build a ~5,000-site website URL list.

Sources:
  1. Y Combinator company directory (ycombinator.com/companies)
  2. Product Hunt top products
  3. Awwwards awarded sites
  4. Godly.website curated designs

Usage:
    python scripts/00_build_seeds.py [--out data/sources/website_urls_large.csv] [--target 5000]
"""

import argparse
import csv
import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl
from pathlib import Path
from collections import OrderedDict

# Bypass SSL issues for scraping
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch(url: str, extra_headers: dict | None = None, timeout: int = 30) -> str:
    """Fetch a URL and return the response body as text."""
    hdrs = {**HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_url(url: str) -> str:
    """Normalize a URL to https://domain.tld format for dedup."""
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    # Remove www.
    url = re.sub(r"https?://(www\.)?", "https://", url)
    return url


def extract_domain(url: str) -> str:
    """Extract the bare domain from a URL."""
    parsed = urllib.parse.urlparse(url)
    return (parsed.netloc or parsed.path).replace("www.", "").lower()


# ---------------------------------------------------------------------------
# Source 1: Y Combinator Company Directory
# ---------------------------------------------------------------------------
def scrape_yc(max_pages: int = 50) -> list[tuple[str, str]]:
    """
    Scrape YC company directory via their internal API.
    Returns list of (url, category_hint) tuples.
    """
    print("\n[YC] Scraping Y Combinator directory...")
    results = []
    base = "https://www.ycombinator.com/companies"

    # YC uses an Algolia-backed API. We can hit the page and extract data
    # from the __NEXT_DATA__ JSON, or use their search endpoint.
    # Let's try the Algolia search that their frontend uses.
    algolia_url = "https://45bwzj1sgc-dsn.algolia.net/1/indexes/*/queries"

    page = 0
    while page < max_pages:
        payload = json.dumps({
            "requests": [{
                "indexName": "YCCompany_production",
                "params": f"hitsPerPage=100&page={page}&query="
            }]
        }).encode()

        query_string = urllib.parse.urlencode({
            "x-algolia-agent": "Algolia for JavaScript (4.14.3); Browser",
            "x-algolia-api-key": "MjBjYjRiMzY0NzdhZWY0NjExY2NhZjYxMGIxYjc2MTAwNWFkNTkwNTc4NjgxYjU0YzFhYTY2ZGQ5OGY5NDMxZnJlc3RyaWN0SW5kaWNlcz0lNUIlMjJZQ0NvbXBhbnlfcHJvZHVjdGlvbiUyMiUyQyUyMllDQ29tcGFueV9CeV9MYXVuY2hfRGF0ZV9wcm9kdWN0aW9uJTIyJTVEJnRhZz1ZQ19Ub3BfQ29tcGFuaWVz",
            "x-algolia-application-id": "45BWZJ1SGC",
        })

        req = urllib.request.Request(
            f"{algolia_url}?{query_string}",
            data=payload,
            headers={**HEADERS, "Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  [YC] Algolia request failed at page {page}: {e}")
            break

        hits = data.get("results", [{}])[0].get("hits", [])
        if not hits:
            break

        for hit in hits:
            url = hit.get("website") or hit.get("url") or ""
            if not url or "ycombinator.com" in url:
                continue
            # Use the industry tags or "Top Company" tag as category
            industries = hit.get("industries", [])
            batch = hit.get("batch", "")
            category = industries[0] if industries else "yc_startup"
            # Clean category
            category = category.lower().replace(" ", "_").replace("&", "and")
            results.append((normalize_url(url), category))

        print(f"  [YC] Page {page}: got {len(hits)} companies (total so far: {len(results)})")
        page += 1
        time.sleep(0.3)

    print(f"  [YC] Total: {len(results)} companies")
    return results


# ---------------------------------------------------------------------------
# Source 2: Product Hunt (via their public GraphQL endpoint)
# ---------------------------------------------------------------------------
def scrape_producthunt(num_days: int = 60, per_day: int = 50) -> list[tuple[str, str]]:
    """
    Scrape Product Hunt top products via their website's embedded data.
    Falls back to scraping the time-travel pages.
    """
    print("\n[PH] Scraping Product Hunt...")
    results = []

    # Product Hunt has a public-facing GraphQL API embedded in their frontend.
    # We'll use the /topics endpoint or the leaderboard pages.
    topics = [
        ("developer-tools", "devtools"),
        ("saas", "saas"),
        ("artificial-intelligence", "ai"),
        ("design-tools", "design"),
        ("productivity", "productivity"),
        ("fintech", "fintech"),
        ("marketing", "marketing"),
        ("e-commerce", "ecommerce"),
        ("health-fitness", "health"),
        ("education", "education"),
        ("social-media", "social"),
        ("analytics", "analytics"),
        ("no-code", "nocode"),
        ("crypto-web3", "web3"),
        ("open-source", "opensource"),
        ("messaging", "messaging"),
        ("video", "video"),
        ("music", "music"),
        ("travel", "travel"),
        ("food-drink", "food"),
    ]

    for topic_slug, category in topics:
        url = f"https://www.producthunt.com/topics/{topic_slug}"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  [PH] Failed to fetch topic {topic_slug}: {e}")
            time.sleep(1)
            continue

        # Extract product URLs from the HTML
        # PH product pages link to external sites via data attributes or redirect URLs
        # Look for website links in the rendered content
        # Pattern: "website":"https://..." in JSON-LD or __NEXT_DATA__
        next_data_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if next_data_match:
            try:
                next_data = json.loads(next_data_match.group(1))
                # Navigate the nested structure to find product URLs
                props = next_data.get("props", {}).get("pageProps", {})
                # Try to find products in various possible paths
                for key in ["topic", "data"]:
                    if key in props:
                        _extract_ph_products(props[key], results, category)
            except (json.JSONDecodeError, KeyError):
                pass

        # Also try regex extraction of external URLs from the page
        # PH pages often have og:url or website links
        ext_urls = re.findall(r'https?://(?!www\.producthunt\.com)[a-zA-Z0-9._-]+\.[a-z]{2,}(?:/[^\s"<>]*)?', html)
        for ext_url in ext_urls:
            domain = extract_domain(ext_url)
            # Filter out common non-product domains
            skip = ["twitter.com", "facebook.com", "google.com", "youtube.com",
                     "linkedin.com", "instagram.com", "github.com", "apple.com",
                     "play.google.com", "cdn.", "cloudfront.", "amazonaws.",
                     "cloudflare.", "fonts.googleapis", "schema.org", "w3.org",
                     "gstatic.com", "googleapis.com"]
            if any(s in domain for s in skip):
                continue
            if len(domain) > 5:  # Skip very short/invalid domains
                results.append((normalize_url(ext_url.split("?")[0].split("#")[0]), category))

        print(f"  [PH] Topic '{topic_slug}': running total {len(results)}")
        time.sleep(1.5)

    print(f"  [PH] Total: {len(results)} products")
    return results


def _extract_ph_products(obj, results, category):
    """Recursively extract product website URLs from PH JSON data."""
    if isinstance(obj, dict):
        if "website" in obj and isinstance(obj["website"], str):
            url = obj["website"]
            if url.startswith("http") and "producthunt.com" not in url:
                results.append((normalize_url(url), category))
        for v in obj.values():
            _extract_ph_products(v, results, category)
    elif isinstance(obj, list):
        for item in obj:
            _extract_ph_products(item, results, category)


# ---------------------------------------------------------------------------
# Source 3: Awwwards (scrape site listings)
# ---------------------------------------------------------------------------
def scrape_awwwards(max_pages: int = 80) -> list[tuple[str, str]]:
    """
    Scrape Awwwards awarded/nominated sites.
    """
    print("\n[AWWWARDS] Scraping Awwwards...")
    results = []

    for page in range(1, max_pages + 1):
        url = f"https://www.awwwards.com/websites/?page={page}"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  [AWWWARDS] Failed page {page}: {e}")
            time.sleep(2)
            continue

        # Awwwards lists sites with links to the actual websites
        # Pattern: data-url="https://..." or class="js-visit" href="..."
        urls_found = re.findall(r'data-url="(https?://[^"]+)"', html)
        if not urls_found:
            # Try alternate pattern
            urls_found = re.findall(r'href="(https?://(?!www\.awwwards\.com)[^"]+)"', html)

        for site_url in urls_found:
            domain = extract_domain(site_url)
            skip = ["awwwards.com", "twitter.com", "facebook.com", "google.com",
                     "youtube.com", "linkedin.com", "instagram.com", "vimeo.com",
                     "cdn.", "cloudfront.", "fonts.", "schema.org"]
            if any(s in domain for s in skip):
                continue
            if len(domain) > 5:
                results.append((normalize_url(site_url.split("?")[0].split("#")[0]), "design_awarded"))

        if not urls_found:
            print(f"  [AWWWARDS] No URLs found on page {page}, stopping.")
            break

        print(f"  [AWWWARDS] Page {page}: found {len(urls_found)} URLs (total: {len(results)})")
        time.sleep(1.5)

    print(f"  [AWWWARDS] Total: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Source 4: Curated tech company lists (manually seeded well-known companies)
# ---------------------------------------------------------------------------
def curated_tech_companies() -> list[tuple[str, str]]:
    """
    A hand-curated list of notable tech companies with good UI/UX.
    These serve as a baseline to ensure coverage of major modern sites.
    """
    print("\n[CURATED] Adding curated tech company list...")
    companies = [
        # AI/ML
        ("https://openai.com", "ai"), ("https://anthropic.com", "ai"),
        ("https://huggingface.co", "ai"), ("https://midjourney.com", "ai"),
        ("https://stability.ai", "ai"), ("https://replicate.com", "ai"),
        ("https://runway.ml", "ai"), ("https://jasper.ai", "ai"),
        ("https://copy.ai", "ai"), ("https://perplexity.ai", "ai"),
        ("https://cohere.com", "ai"), ("https://anyscale.com", "ai"),
        ("https://weights.gg", "ai"), ("https://together.ai", "ai"),
        ("https://modal.com", "ai"), ("https://deepmind.google", "ai"),

        # Dev tools / Infrastructure
        ("https://vercel.com", "devtools"), ("https://netlify.com", "devtools"),
        ("https://railway.app", "devtools"), ("https://render.com", "devtools"),
        ("https://fly.io", "devtools"), ("https://supabase.com", "devtools"),
        ("https://planetscale.com", "devtools"), ("https://neon.tech", "devtools"),
        ("https://turso.tech", "devtools"), ("https://convex.dev", "devtools"),
        ("https://clerk.com", "devtools"), ("https://auth0.com", "devtools"),
        ("https://sentry.io", "devtools"), ("https://datadog.com", "devtools"),
        ("https://grafana.com", "devtools"), ("https://postman.com", "devtools"),
        ("https://insomnia.rest", "devtools"), ("https://deno.com", "devtools"),
        ("https://bun.sh", "devtools"), ("https://astro.build", "devtools"),
        ("https://remix.run", "devtools"), ("https://nextjs.org", "devtools"),
        ("https://svelte.dev", "devtools"), ("https://tailwindcss.com", "devtools"),
        ("https://prisma.io", "devtools"), ("https://drizzle.team", "devtools"),
        ("https://temporal.io", "devtools"), ("https://dagster.io", "devtools"),
        ("https://prefect.io", "devtools"), ("https://pulumi.com", "devtools"),
        ("https://terraform.io", "devtools"), ("https://docker.com", "devtools"),
        ("https://circleci.com", "devtools"), ("https://buildkite.com", "devtools"),
        ("https://launchdarkly.com", "devtools"), ("https://split.io", "devtools"),
        ("https://cockroachlabs.com", "devtools"), ("https://timescale.com", "devtools"),
        ("https://clickhouse.com", "devtools"), ("https://materialize.com", "devtools"),
        ("https://dbt.com", "devtools"), ("https://fivetran.com", "devtools"),
        ("https://airbyte.com", "devtools"), ("https://retool.com", "devtools"),
        ("https://appsmith.com", "devtools"), ("https://budibase.com", "devtools"),

        # SaaS / Productivity
        ("https://notion.so", "productivity"), ("https://coda.io", "productivity"),
        ("https://airtable.com", "productivity"), ("https://clickup.com", "productivity"),
        ("https://monday.com", "productivity"), ("https://asana.com", "productivity"),
        ("https://linear.app", "productivity"), ("https://height.app", "productivity"),
        ("https://shortcut.com", "productivity"), ("https://todoist.com", "productivity"),
        ("https://obsidian.md", "productivity"), ("https://craft.do", "productivity"),
        ("https://mem.ai", "productivity"), ("https://roamresearch.com", "productivity"),
        ("https://pitch.com", "productivity"), ("https://gamma.app", "productivity"),
        ("https://tome.app", "productivity"), ("https://miro.com", "productivity"),
        ("https://whimsical.com", "productivity"), ("https://excalidraw.com", "productivity"),
        ("https://loom.com", "productivity"), ("https://grain.com", "productivity"),
        ("https://around.co", "productivity"), ("https://tandem.chat", "productivity"),
        ("https://superhuman.com", "productivity"), ("https://front.com", "productivity"),
        ("https://missiveapp.com", "productivity"), ("https://spike.email", "productivity"),
        ("https://calendly.com", "productivity"), ("https://savvycal.com", "productivity"),
        ("https://cal.com", "productivity"), ("https://reclaim.ai", "productivity"),
        ("https://slack.com", "productivity"), ("https://discord.com", "productivity"),

        # Design
        ("https://figma.com", "design"), ("https://framer.com", "design"),
        ("https://webflow.com", "design"), ("https://sketch.com", "design"),
        ("https://zeplin.io", "design"), ("https://storybook.js.org", "design"),
        ("https://rive.app", "design"), ("https://spline.design", "design"),
        ("https://lottiefiles.com", "design"), ("https://iconify.design", "design"),
        ("https://heroicons.com", "design"), ("https://phosphoricons.com", "design"),
        ("https://coolors.co", "design"), ("https://colorhunt.co", "design"),
        ("https://fontjoy.com", "design"), ("https://typescale.com", "design"),

        # Fintech
        ("https://stripe.com", "fintech"), ("https://plaid.com", "fintech"),
        ("https://mercury.com", "fintech"), ("https://brex.com", "fintech"),
        ("https://ramp.com", "fintech"), ("https://wise.com", "fintech"),
        ("https://revolut.com", "fintech"), ("https://robinhood.com", "fintech"),
        ("https://coinbase.com", "fintech"), ("https://kraken.com", "fintech"),
        ("https://carta.com", "fintech"), ("https://gusto.com", "fintech"),
        ("https://rippling.com", "fintech"), ("https://deel.com", "fintech"),
        ("https://remote.com", "fintech"), ("https://pilot.com", "fintech"),
        ("https://bench.co", "fintech"), ("https://wave.com", "fintech"),
        ("https://affirm.com", "fintech"), ("https://klarna.com", "fintech"),

        # E-commerce / Consumer
        ("https://shopify.com", "ecommerce"), ("https://gumroad.com", "ecommerce"),
        ("https://lemonsqueezy.com", "ecommerce"), ("https://paddle.com", "ecommerce"),
        ("https://bigcommerce.com", "ecommerce"), ("https://wix.com", "ecommerce"),
        ("https://squarespace.com", "ecommerce"), ("https://carrd.co", "ecommerce"),
        ("https://typedream.com", "ecommerce"), ("https://super.so", "ecommerce"),
        ("https://podia.com", "ecommerce"), ("https://teachable.com", "ecommerce"),
        ("https://thinkific.com", "ecommerce"), ("https://kajabi.com", "ecommerce"),
        ("https://convertkit.com", "ecommerce"), ("https://beehiiv.com", "ecommerce"),
        ("https://substack.com", "ecommerce"), ("https://ghost.org", "ecommerce"),

        # Marketing / Growth
        ("https://hubspot.com", "marketing"), ("https://mailchimp.com", "marketing"),
        ("https://sendgrid.com", "marketing"), ("https://twilio.com", "marketing"),
        ("https://segment.com", "marketing"), ("https://amplitude.com", "marketing"),
        ("https://mixpanel.com", "marketing"), ("https://posthog.com", "marketing"),
        ("https://heap.io", "marketing"), ("https://hotjar.com", "marketing"),
        ("https://intercom.com", "marketing"), ("https://zendesk.com", "marketing"),
        ("https://freshworks.com", "marketing"), ("https://drift.com", "marketing"),
        ("https://crisp.chat", "marketing"), ("https://Customer.io", "marketing"),
        ("https://braze.com", "marketing"), ("https://iterable.com", "marketing"),
        ("https://attio.com", "marketing"), ("https://clay.com", "marketing"),
        ("https://apollo.io", "marketing"), ("https://clearbit.com", "marketing"),

        # Security / Infra
        ("https://1password.com", "security"), ("https://bitwarden.com", "security"),
        ("https://crowdstrike.com", "security"), ("https://snyk.io", "security"),
        ("https://hashicorp.com", "security"), ("https://tailscale.com", "security"),
        ("https://cloudflare.com", "security"), ("https://fastly.com", "security"),
        ("https://akamai.com", "security"), ("https://zscaler.com", "security"),
        ("https://wiz.io", "security"), ("https://orca.security", "security"),

        # Consumer apps with great UI
        ("https://spotify.com", "consumer"), ("https://airbnb.com", "consumer"),
        ("https://uber.com", "consumer"), ("https://lyft.com", "consumer"),
        ("https://doordash.com", "consumer"), ("https://instacart.com", "consumer"),
        ("https://tiktok.com", "consumer"), ("https://snapchat.com", "consumer"),
        ("https://pinterest.com", "consumer"), ("https://reddit.com", "consumer"),
        ("https://twitch.tv", "consumer"), ("https://netflix.com", "consumer"),
        ("https://hulu.com", "consumer"), ("https://disneyplus.com", "consumer"),
        ("https://duolingo.com", "consumer"), ("https://headspace.com", "consumer"),
        ("https://calm.com", "consumer"), ("https://peloton.com", "consumer"),
        ("https://strava.com", "consumer"), ("https://alltrails.com", "consumer"),
        ("https://bumble.com", "consumer"), ("https://hinge.co", "consumer"),

        # Health / Bio
        ("https://ro.co", "health"), ("https://hims.com", "health"),
        ("https://nurx.com", "health"), ("https://cerebral.com", "health"),
        ("https://oura.com", "health"), ("https://whoop.com", "health"),
        ("https://withings.com", "health"), ("https://levels.link", "health"),
        ("https://tempus.com", "health"),

        # Real estate / Proptech
        ("https://opendoor.com", "proptech"), ("https://compass.com", "proptech"),
        ("https://zillow.com", "proptech"), ("https://redfin.com", "proptech"),
        ("https://latch.com", "proptech"),

        # Education
        ("https://coursera.org", "education"), ("https://udemy.com", "education"),
        ("https://brilliant.org", "education"), ("https://khanacademy.org", "education"),
        ("https://codecademy.com", "education"), ("https://replit.com", "education"),
        ("https://codepen.io", "education"), ("https://codesandbox.io", "education"),
        ("https://hashnode.com", "education"), ("https://dev.to", "education"),

        # Agency / Portfolio sites (great design)
        ("https://basicagency.com", "agency"), ("https://fantasy.co", "agency"),
        ("https://huge.com", "agency"), ("https://instrument.com", "agency"),
        ("https://metalab.com", "agency"), ("https://ueno.co", "agency"),
        ("https://unfold.co", "agency"), ("https://work.co", "agency"),

        # Additional AI / ML
        ("https://deeplearning.ai", "ai"), ("https://wandb.ai", "ai"),
        ("https://labelbox.com", "ai"), ("https://scale.com", "ai"),
        ("https://datarobot.com", "ai"), ("https://h2o.ai", "ai"),
        ("https://determined.ai", "ai"), ("https://pinecone.io", "ai"),
        ("https://weaviate.io", "ai"), ("https://qdrant.tech", "ai"),
        ("https://chroma.ai", "ai"), ("https://langchain.com", "ai"),
        ("https://llamaindex.ai", "ai"), ("https://unstructured.io", "ai"),
        ("https://dify.ai", "ai"), ("https://flowise.ai", "ai"),
        ("https://character.ai", "ai"), ("https://poe.com", "ai"),
        ("https://you.com", "ai"), ("https://phind.com", "ai"),
        ("https://cursor.com", "ai"), ("https://codeium.com", "ai"),
        ("https://tabnine.com", "ai"), ("https://sourcegraph.com", "ai"),
        ("https://continue.dev", "ai"), ("https://pieces.app", "ai"),

        # More devtools
        ("https://encore.dev", "devtools"), ("https://buf.build", "devtools"),
        ("https://connectrpc.com", "devtools"), ("https://trpc.io", "devtools"),
        ("https://graphql.org", "devtools"), ("https://hasura.io", "devtools"),
        ("https://stellate.co", "devtools"), ("https://redpanda.com", "devtools"),
        ("https://upstash.com", "devtools"), ("https://inngest.com", "devtools"),
        ("https://trigger.dev", "devtools"), ("https://qstash.upstash.com", "devtools"),
        ("https://resend.com", "devtools"), ("https://loops.so", "devtools"),
        ("https://plunk.dev", "devtools"), ("https://knock.app", "devtools"),
        ("https://novu.co", "devtools"), ("https://magicbell.com", "devtools"),
        ("https://permit.io", "devtools"), ("https://cerbos.dev", "devtools"),
        ("https://oso.dev", "devtools"), ("https://sniptt.com", "devtools"),
        ("https://depot.dev", "devtools"), ("https://earthly.dev", "devtools"),
        ("https://dagger.io", "devtools"), ("https://flightcontrol.dev", "devtools"),
        ("https://sst.dev", "devtools"), ("https://seed.run", "devtools"),
        ("https://serverless.com", "devtools"), ("https://val.town", "devtools"),
        ("https://workers.cloudflare.com", "devtools"), ("https://fermyon.com", "devtools"),
        ("https://wasmer.io", "devtools"), ("https://wasmcloud.com", "devtools"),
        ("https://turbo.build", "devtools"), ("https://nx.dev", "devtools"),
        ("https://moonrepo.dev", "devtools"), ("https://bazel.build", "devtools"),
        ("https://biomejs.dev", "devtools"), ("https://oxc-project.github.io", "devtools"),
        ("https://rome.tools", "devtools"), ("https://vitest.dev", "devtools"),
        ("https://playwright.dev", "devtools"), ("https://cypress.io", "devtools"),
        ("https://chromatic.com", "devtools"), ("https://percy.io", "devtools"),
        ("https://backstage.io", "devtools"), ("https://port.io", "devtools"),
        ("https://cortex.io", "devtools"), ("https://getdx.com", "devtools"),
        ("https://linearb.io", "devtools"), ("https://swarmia.com", "devtools"),
        ("https://gitpod.io", "devtools"), ("https://coder.com", "devtools"),
        ("https://stackblitz.com", "devtools"), ("https://glitch.com", "devtools"),

        # More SaaS / business
        ("https://lattice.com", "hr"), ("https://lever.co", "hr"),
        ("https://greenhouse.io", "hr"), ("https://ashbyhq.com", "hr"),
        ("https://gem.com", "hr"), ("https://dover.com", "hr"),
        ("https://wellfound.com", "hr"), ("https://triplebyte.com", "hr"),
        ("https://vettery.com", "hr"), ("https://hired.com", "hr"),
        ("https://gong.io", "sales"), ("https://chorus.ai", "sales"),
        ("https://outreach.io", "sales"), ("https://salesloft.com", "sales"),
        ("https://chilipiper.com", "sales"), ("https://qualified.com", "sales"),
        ("https://6sense.com", "sales"), ("https://demandbase.com", "sales"),
        ("https://bombora.com", "sales"), ("https://zoominfo.com", "sales"),
        ("https://lusha.com", "sales"), ("https://seamless.ai", "sales"),
        ("https://freshsales.io", "sales"), ("https://close.com", "sales"),
        ("https://pipedrive.com", "sales"), ("https://copper.com", "sales"),
        ("https://streak.com", "sales"), ("https://folk.app", "sales"),

        # Legal tech
        ("https://ironclad.com", "legaltech"), ("https://contractbook.com", "legaltech"),
        ("https://juro.com", "legaltech"), ("https://clio.com", "legaltech"),
        ("https://notion.so/legal", "legaltech"), ("https://harvey.ai", "legaltech"),

        # Data / Analytics
        ("https://snowflake.com", "data"), ("https://databricks.com", "data"),
        ("https://dbt.com", "data"), ("https://metabase.com", "data"),
        ("https://superset.apache.org", "data"), ("https://hex.tech", "data"),
        ("https://mode.com", "data"), ("https://sigma.com", "data"),
        ("https://lightdash.com", "data"), ("https://census.com", "data"),
        ("https://hightouch.com", "data"), ("https://rudderstack.com", "data"),
        ("https://meltano.com", "data"), ("https://dagster.io", "data"),
        ("https://astronomer.io", "data"), ("https://prefect.io", "data"),
        ("https://tecton.ai", "data"), ("https://feast.dev", "data"),

        # Crypto / Web3
        ("https://ethereum.org", "web3"), ("https://solana.com", "web3"),
        ("https://polygon.technology", "web3"), ("https://arbitrum.io", "web3"),
        ("https://optimism.io", "web3"), ("https://base.org", "web3"),
        ("https://uniswap.org", "web3"), ("https://aave.com", "web3"),
        ("https://lido.fi", "web3"), ("https://metamask.io", "web3"),
        ("https://rainbow.me", "web3"), ("https://phantom.app", "web3"),
        ("https://opensea.io", "web3"), ("https://blur.io", "web3"),
        ("https://zora.co", "web3"), ("https://mirror.xyz", "web3"),
        ("https://farcaster.xyz", "web3"), ("https://lens.xyz", "web3"),
        ("https://worldcoin.org", "web3"), ("https://alchemy.com", "web3"),
        ("https://infura.io", "web3"), ("https://thirdweb.com", "web3"),
        ("https://moralis.io", "web3"),

        # Climate / Clean tech
        ("https://watershed.com", "climate"), ("https://patch.io", "climate"),
        ("https://persefoni.com", "climate"), ("https://pachama.com", "climate"),
        ("https://wren.co", "climate"), ("https://charm.io", "climate"),

        # More consumer / lifestyle
        ("https://warbyparker.com", "consumer"), ("https://glossier.com", "consumer"),
        ("https://everlane.com", "consumer"), ("https://allbirds.com", "consumer"),
        ("https://casper.com", "consumer"), ("https://away.com", "consumer"),
        ("https://brooklinen.com", "consumer"), ("https://parachutehome.com", "consumer"),
        ("https://ritual.com", "consumer"), ("https://huel.com", "consumer"),
        ("https://athleticgreens.com", "consumer"), ("https://whoop.com", "consumer"),
        ("https://therabody.com", "consumer"), ("https://hyperice.com", "consumer"),
        ("https://mirror.co", "consumer"), ("https://tonal.com", "consumer"),
        ("https://noom.com", "consumer"), ("https://calibrate.com", "consumer"),
        ("https://ro.co", "consumer"), ("https://keeps.com", "consumer"),
        ("https://hims.com", "consumer"), ("https://nurx.com", "consumer"),
        ("https://getquip.com", "consumer"), ("https://byte.com", "consumer"),
        ("https://candid.com", "consumer"), ("https://smiledirectclub.com", "consumer"),

        # Food / Delivery
        ("https://sweetgreen.com", "food"), ("https://chipotle.com", "food"),
        ("https://blueapron.com", "food"), ("https://hellofresh.com", "food"),
        ("https://factor75.com", "food"), ("https://dailyharvest.com", "food"),
        ("https://territoryfoods.com", "food"), ("https://hungryroot.com", "food"),

        # Media / Publishing with modern design
        ("https://theverge.com", "media"), ("https://wired.com", "media"),
        ("https://arstechnica.com", "media"), ("https://techcrunch.com", "media"),
        ("https://theinformation.com", "media"), ("https://semafor.com", "media"),
        ("https://puck.news", "media"), ("https://defector.com", "media"),
        ("https://every.to", "media"), ("https://stratechery.com", "media"),
        ("https://platformer.news", "media"), ("https://newcomer.co", "media"),
        ("https://restofworld.org", "media"), ("https://protocol.com", "media"),

        # SaaS verticals
        ("https://toast.com", "vertical_saas"), ("https://mindbody.com", "vertical_saas"),
        ("https://servicetitan.com", "vertical_saas"), ("https://procore.com", "vertical_saas"),
        ("https://veeva.com", "vertical_saas"), ("https://benchling.com", "vertical_saas"),
        ("https://blend.com", "vertical_saas"), ("https://ncino.com", "vertical_saas"),
        ("https://qualia.com", "vertical_saas"), ("https://buildium.com", "vertical_saas"),
        ("https://appfolio.com", "vertical_saas"), ("https://yardi.com", "vertical_saas"),
        ("https://athenahealth.com", "vertical_saas"), ("https://elation.com", "vertical_saas"),

        # Communication / Video
        ("https://zoom.us", "communication"), ("https://webex.com", "communication"),
        ("https://gather.town", "communication"), ("https://mmhmm.app", "communication"),
        ("https://riverside.fm", "communication"), ("https://descript.com", "communication"),
        ("https://kapwing.com", "communication"), ("https://canva.com", "communication"),
        ("https://mux.com", "communication"), ("https://cloudinary.com", "communication"),
        ("https://imgix.com", "communication"), ("https://uploadcare.com", "communication"),

        # Misc well-designed sites
        ("https://notion.so", "productivity"), ("https://raycast.com", "productivity"),
        ("https://arc.net", "productivity"), ("https://warp.dev", "productivity"),
        ("https://fig.io", "productivity"), ("https://iterm2.com", "productivity"),
        ("https://hyper.is", "productivity"), ("https://alacritty.org", "productivity"),
        ("https://zed.dev", "productivity"), ("https://lapce.dev", "productivity"),

        # More agencies / studios
        ("https://rally.io", "agency"), ("https://resn.co.nz", "agency"),
        ("https://activetheory.net", "agency"), ("https://bukwild.com", "agency"),
        ("https://playground.xyz", "agency"), ("https://immersive-g.com", "agency"),
        ("https://unit9.com", "agency"), ("https://northkingdom.com", "agency"),
        ("https://dogstudio.co", "agency"), ("https://locomotive.ca", "agency"),
        ("https://cuberto.com", "agency"), ("https://makemepulse.com", "agency"),
        ("https://hello.monday.com", "agency"), ("https://area17.com", "agency"),
        ("https://fivefifty.co", "agency"), ("https://humaan.com", "agency"),
        ("https://14islands.com", "agency"), ("https://littleworkstudio.com", "agency"),
        ("https://adoratorio.studio", "agency"), ("https://aristidebenoist.com", "agency"),

        # Hosting / Cloud
        ("https://digitalocean.com", "cloud"), ("https://linode.com", "cloud"),
        ("https://vultr.com", "cloud"), ("https://hetzner.com", "cloud"),
        ("https://oracle.com/cloud", "cloud"), ("https://heroku.com", "cloud"),
        ("https://northflank.com", "cloud"), ("https://porter.run", "cloud"),
        ("https://coherence.io", "cloud"), ("https://aptible.com", "cloud"),

        # Payments / Billing
        ("https://chargebee.com", "billing"), ("https://recurly.com", "billing"),
        ("https://chargify.com", "billing"), ("https://lago.dev", "billing"),
        ("https://getlago.com", "billing"), ("https://metronome.com", "billing"),
        ("https://orb.com", "billing"), ("https://stigg.io", "billing"),

        # Auth / Identity
        ("https://stytch.com", "auth"), ("https://workos.com", "auth"),
        ("https://descope.com", "auth"), ("https://frontegg.com", "auth"),
        ("https://propelauth.com", "auth"), ("https://hanko.io", "auth"),
        ("https://supertokens.com", "auth"), ("https://fusionauth.io", "auth"),

        # Open source project sites with good design
        ("https://vuejs.org", "opensource"), ("https://angular.dev", "opensource"),
        ("https://solidjs.com", "opensource"), ("https://qwik.builder.io", "opensource"),
        ("https://htmx.org", "opensource"), ("https://elixir-lang.org", "opensource"),
        ("https://gleam.run", "opensource"), ("https://ziglang.org", "opensource"),
        ("https://rust-lang.org", "opensource"), ("https://go.dev", "opensource"),
        ("https://kotlinlang.org", "opensource"), ("https://swift.org", "opensource"),
        ("https://typescriptlang.org", "opensource"), ("https://python.org", "opensource"),
        ("https://ruby-lang.org", "opensource"), ("https://pnpm.io", "opensource"),
        ("https://yarnpkg.com", "opensource"), ("https://esbuild.github.io", "opensource"),
        ("https://rollupjs.org", "opensource"), ("https://parceljs.org", "opensource"),
        ("https://webpack.js.org", "opensource"), ("https://vitejs.dev", "opensource"),
    ]
    print(f"  [CURATED] {len(companies)} companies")
    return companies


# ---------------------------------------------------------------------------
# Source 5: Scrape Godly.website
# ---------------------------------------------------------------------------
def scrape_godly(max_pages: int = 40) -> list[tuple[str, str]]:
    """Scrape Godly.website for curated modern web designs."""
    print("\n[GODLY] Scraping Godly.website...")
    results = []

    # Godly loads via JS, but we can try to get data from their page
    try:
        html = fetch("https://godly.website")
        # Extract external URLs from the page source
        urls = re.findall(r'href="(https?://(?!godly\.website)[^"]+)"', html)
        for url in urls:
            domain = extract_domain(url)
            skip = ["twitter.com", "facebook.com", "google.com", "youtube.com",
                     "linkedin.com", "instagram.com", "cdn.", "fonts.", "schema.org",
                     "w3.org", "gstatic.com", "googleapis.com", "x.com"]
            if any(s in domain for s in skip):
                continue
            if len(domain) > 5:
                results.append((normalize_url(url.split("?")[0].split("#")[0]), "design_curated"))
    except Exception as e:
        print(f"  [GODLY] Failed: {e}")

    print(f"  [GODLY] Total: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Source 6: SaaS design inspiration sites
# ---------------------------------------------------------------------------
def scrape_saas_pages(max_pages: int = 30) -> list[tuple[str, str]]:
    """Scrape SaaS landing page galleries."""
    print("\n[SAAS] Scraping SaaS landing page galleries...")
    results = []

    # Try saaslandingpage.com
    for page in range(1, max_pages + 1):
        url = f"https://saaslandingpage.com/page/{page}/" if page > 1 else "https://saaslandingpage.com/"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  [SAAS] Failed page {page}: {e}")
            time.sleep(1)
            break

        # Extract external URLs
        urls = re.findall(r'href="(https?://(?!saaslandingpage\.com)[^"]+)"', html)
        page_results = 0
        for ext_url in urls:
            domain = extract_domain(ext_url)
            skip = ["twitter.com", "facebook.com", "google.com", "youtube.com",
                     "linkedin.com", "instagram.com", "cdn.", "fonts.", "schema.org",
                     "w3.org", "gstatic.com", "googleapis.com", "x.com",
                     "wordpress.org", "wp.com", "gravatar.com"]
            if any(s in domain for s in skip):
                continue
            if len(domain) > 5:
                results.append((normalize_url(ext_url.split("?")[0].split("#")[0]), "saas"))
                page_results += 1

        if page_results == 0:
            break
        print(f"  [SAAS] Page {page}: {page_results} URLs (total: {len(results)})")
        time.sleep(1.0)

    # Try land-book.com
    for page in range(1, 20):
        url = f"https://land-book.com/websites?page={page}"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  [LANDBOOK] Failed page {page}: {e}")
            time.sleep(1)
            break

        urls = re.findall(r'href="(https?://(?!land-book\.com)[^"]+)"', html)
        page_results = 0
        for ext_url in urls:
            domain = extract_domain(ext_url)
            skip = ["twitter.com", "facebook.com", "google.com", "youtube.com",
                     "linkedin.com", "instagram.com", "cdn.", "fonts.", "schema.org",
                     "w3.org", "gstatic.com", "googleapis.com", "x.com",
                     "cloudflare.com", "jquery.com"]
            if any(s in domain for s in skip):
                continue
            if len(domain) > 5:
                results.append((normalize_url(ext_url.split("?")[0].split("#")[0]), "design_landing"))
                page_results += 1

        if page_results == 0:
            break
        print(f"  [LANDBOOK] Page {page}: {page_results} URLs (total: {len(results)})")
        time.sleep(1.0)

    print(f"  [SAAS] Total: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Deduplication and output
# ---------------------------------------------------------------------------
def deduplicate(all_results: list[tuple[str, str]], existing_csv: Path | None = None) -> list[tuple[str, str]]:
    """Deduplicate by domain, preserving first-seen category."""
    seen_domains = set()
    deduped = OrderedDict()

    # Load existing website URLs first so they take priority
    if existing_csv and existing_csv.exists():
        with open(existing_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = normalize_url(row["url"])
                domain = extract_domain(url)
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    deduped[domain] = (url, row.get("category_hint", ""))

    for url, category in all_results:
        domain = extract_domain(url)
        if domain not in seen_domains and domain:
            seen_domains.add(domain)
            # Ensure the URL points to the root of the site
            parsed = urllib.parse.urlparse(url)
            root_url = f"https://{parsed.netloc}"
            deduped[domain] = (root_url, category)

    return list(deduped.values())


def main():
    parser = argparse.ArgumentParser(description="Build website URL list from curated sources")
    parser.add_argument("--out", default="data/sources/website_urls_large.csv", help="Output CSV path")
    parser.add_argument("--target", type=int, default=5000, help="Target number of URLs")
    parser.add_argument(
        "--existing",
        default="data/sources/website_urls.csv",
        help="Existing website URL CSV to preserve",
    )
    parser.add_argument("--skip-scrape", action="store_true", help="Only use curated list (no web scraping)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = root / args.out
    existing_path = root / args.existing if args.existing else None

    all_results = []

    # Always include curated companies (guaranteed quality)
    all_results.extend(curated_tech_companies())

    if not args.skip_scrape:
        # Scrape additional sources
        try:
            all_results.extend(scrape_yc(max_pages=40))
        except Exception as e:
            print(f"  [YC] Failed entirely: {e}")

        try:
            all_results.extend(scrape_awwwards(max_pages=120))
        except Exception as e:
            print(f"  [AWWWARDS] Failed entirely: {e}")

        try:
            all_results.extend(scrape_producthunt(num_days=60))
        except Exception as e:
            print(f"  [PH] Failed entirely: {e}")

        try:
            all_results.extend(scrape_godly(max_pages=40))
        except Exception as e:
            print(f"  [GODLY] Failed entirely: {e}")

        try:
            all_results.extend(scrape_saas_pages(max_pages=30))
        except Exception as e:
            print(f"  [SAAS] Failed entirely: {e}")

    # Deduplicate
    deduped = deduplicate(all_results, existing_path)
    print(f"\n{'='*60}")
    print(f"Total unique sites after dedup: {len(deduped)}")

    if len(deduped) < args.target:
        print(f"  Warning: only got {len(deduped)}/{args.target} target URLs")
        print(f"  Consider adding more sources or increasing page limits")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "category_hint"])
        for url, category in deduped:
            writer.writerow([url, category])

    print(f"\nWritten to: {out_path}")
    print(f"Total URLs: {len(deduped)}")

    # Print breakdown by category
    from collections import Counter
    cats = Counter(cat for _, cat in deduped)
    print(f"\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat:25s} {count:5d}")


if __name__ == "__main__":
    main()
