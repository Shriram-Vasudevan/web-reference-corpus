"""Claude VLM-based style labeling for website clusters."""

import base64
import json
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import config
from src.utils.prompt_templates import CLUSTER_LABEL_SYSTEM, CLUSTER_LABEL_USER

load_dotenv()


def _encode_image(path: str | Path) -> str:
    """Read an image file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _build_messages(screenshot_paths: list[str]) -> list[dict]:
    """Build the messages array with base64-encoded images."""
    content = []
    for path in screenshot_paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image(path),
            },
        })

    content.append({
        "type": "text",
        "text": CLUSTER_LABEL_USER.format(count=len(screenshot_paths)),
    })

    return [{"role": "user", "content": content}]


def label_cluster(
    screenshot_paths: list[str],
    max_samples: int = config.LABEL_SAMPLES_PER_CLUSTER,
    max_retries: int = config.LABEL_MAX_RETRIES,
    dry_run: bool = False,
) -> tuple[dict, str]:
    """Label a cluster using Claude vision.

    Args:
        screenshot_paths: Paths to screenshot images in this cluster.
        max_samples: Max images to send (randomly sampled if more).
        max_retries: Retries on JSON parse failure.
        dry_run: If True, return the prompt without calling the API.

    Returns:
        (parsed_label_dict, raw_response_text)
    """
    # Sample if needed
    if len(screenshot_paths) > max_samples:
        paths = random.sample(screenshot_paths, max_samples)
    else:
        paths = screenshot_paths

    messages = _build_messages(paths)

    if dry_run:
        # Return the structure without calling API
        text_parts = [c for c in messages[0]["content"] if c["type"] == "text"]
        prompt_text = text_parts[0]["text"] if text_parts else ""
        return {
            "dry_run": True,
            "n_images": len(paths),
            "prompt_preview": prompt_text[:500],
        }, ""

    client = anthropic.Anthropic()

    for attempt in range(1, max_retries + 1):
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=CLUSTER_LABEL_SYSTEM,
            messages=messages,
        )

        raw_text = response.content[0].text.strip()

        # Try to extract JSON from response
        try:
            # Handle potential markdown wrapping
            json_text = raw_text
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                json_text = "\n".join(lines[1:-1])
            parsed = json.loads(json_text)

            # Validate required fields
            required = ["page_type", "visual_style", "quality_score"]
            if all(k in parsed for k in required):
                # Normalize distinguishing_features to comma-separated string if list
                if isinstance(parsed.get("distinguishing_features"), list):
                    parsed["distinguishing_features"] = ", ".join(parsed["distinguishing_features"])
                # Ensure quality_score is int
                parsed["quality_score"] = int(parsed["quality_score"])
                return parsed, raw_text

        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt < max_retries:
                continue
            # Return best-effort on final attempt
            return {"page_type": "parse_error", "visual_style": "unknown", "quality_score": 0, "raw": raw_text}, raw_text

    return {"page_type": "error", "visual_style": "unknown", "quality_score": 0}, ""
