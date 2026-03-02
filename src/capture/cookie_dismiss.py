"""Cookie banner dismissal utilities."""

# Common selectors for cookie consent buttons
COOKIE_SELECTORS = [
    # Generic accept buttons
    "button[id*='accept' i]",
    "button[class*='accept' i]",
    "a[id*='accept' i]",
    "button[id*='agree' i]",
    "button[class*='agree' i]",
    # Common cookie consent frameworks
    "[data-testid='cookie-policy-manage-dialog-accept-button']",
    "#onetrust-accept-btn-handler",
    ".onetrust-accept-btn-handler",
    "[class*='cookie'] button[class*='accept' i]",
    "[class*='cookie'] button[class*='allow' i]",
    "[class*='consent'] button[class*='accept' i]",
    "[id*='cookie'] button",
    ".cc-accept",
    ".cc-btn.cc-dismiss",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "[data-cookiefirst-action='accept']",
    ".js-cookie-consent-agree",
    # Text-based matching
    "button:has-text('Accept')",
    "button:has-text('Accept All')",
    "button:has-text('Accept all')",
    "button:has-text('Got it')",
    "button:has-text('I agree')",
    "button:has-text('OK')",
    "button:has-text('Allow')",
    "button:has-text('Allow all')",
]

# CSS to hide common cookie banner containers as fallback
COOKIE_HIDE_CSS = """
[class*="cookie" i],
[id*="cookie" i],
[class*="consent" i],
[id*="consent" i],
[class*="gdpr" i],
[id*="gdpr" i],
.cc-window,
#onetrust-banner-sdk,
.CookieConsent,
[aria-label*="cookie" i] {
    display: none !important;
    visibility: hidden !important;
}
"""


async def dismiss_cookies(page) -> bool:
    """Try to dismiss cookie banners. Returns True if a banner was dismissed."""
    for selector in COOKIE_SELECTORS:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=500):
                await btn.click(timeout=2000)
                return True
        except Exception:
            continue
    return False


async def inject_cookie_css(page):
    """Inject CSS to hide cookie banners as fallback."""
    await page.add_style_tag(content=COOKIE_HIDE_CSS)
