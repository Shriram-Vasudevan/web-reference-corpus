"""Playwright-based website screenshot engine."""

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

import config
from src.capture.cookie_dismiss import dismiss_cookies, inject_cookie_css


def url_to_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    safe = domain.replace(".", "_").replace(":", "_")
    return f"{safe}.png"


async def capture_screenshot(
    url: str,
    output_dir: Path = config.SCREENSHOTS_DIR,
    browser_context=None,
) -> Path:
    """Capture a viewport screenshot of a single URL.

    Args:
        url: The URL to capture.
        output_dir: Directory to save the screenshot.
        browser_context: Optional existing browser context to reuse.

    Returns:
        Path to the saved screenshot.

    Raises:
        Exception: If screenshot capture fails after retries.
    """
    output_path = output_dir / url_to_filename(url)
    owns_context = browser_context is None

    if owns_context:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        browser_context = await browser.new_context(
            viewport={"width": config.VIEWPORT_WIDTH, "height": config.VIEWPORT_HEIGHT},
            device_scale_factor=1,
        )

    try:
        page = await browser_context.new_page()
        try:
            # Navigate with networkidle wait
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=config.SCREENSHOT_TIMEOUT_MS,
            )

            # Dismiss cookie banners
            dismissed = await dismiss_cookies(page)
            if not dismissed:
                await inject_cookie_css(page)

            # Settle time for SPAs
            await asyncio.sleep(config.SETTLE_TIME_S)

            # Viewport-only screenshot (not full page)
            await page.screenshot(path=str(output_path), full_page=False)
            return output_path

        finally:
            await page.close()
    finally:
        if owns_context:
            await browser_context.browser.close()
            await pw.stop()


async def capture_batch(
    urls: list[dict],
    output_dir: Path = config.SCREENSHOTS_DIR,
    max_concurrent: int = config.MAX_CONCURRENT_BROWSERS,
    retries: int = config.CAPTURE_RETRIES,
    on_success=None,
    on_failure=None,
):
    """Capture screenshots for a batch of URLs with concurrency control.

    Args:
        urls: List of dicts with 'url', 'site_id', and optional 'category_hint'.
        output_dir: Directory to save screenshots.
        max_concurrent: Max concurrent browser contexts.
        retries: Number of retry attempts per URL.
        on_success: Callback(site_id, path) on successful capture.
        on_failure: Callback(site_id, error_str) on failed capture.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _capture_one(item: dict):
        url = item["url"]
        site_id = item["site_id"]

        async with semaphore:
            last_error = None
            for attempt in range(1, retries + 1):
                try:
                    pw = await async_playwright().start()
                    browser = await pw.chromium.launch(headless=True)
                    context = await browser.new_context(
                        viewport={
                            "width": config.VIEWPORT_WIDTH,
                            "height": config.VIEWPORT_HEIGHT,
                        },
                        device_scale_factor=1,
                    )
                    try:
                        path = await capture_screenshot(
                            url, output_dir, browser_context=context
                        )
                        if on_success:
                            on_success(site_id, str(path))
                        return
                    finally:
                        await browser.close()
                        await pw.stop()
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        await asyncio.sleep(2 * attempt)

            if on_failure:
                on_failure(site_id, str(last_error))

    await asyncio.gather(*[_capture_one(item) for item in urls])
