#!/usr/bin/env python3
"""Evaluate baseline vs reference-backed generation outputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "claude-sonnet-4-20250514"
OUTPUT_KEYS = ("output", "response", "generated", "code", "text")


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    baseline_output: str
    reference_output: str


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return payload["items"]
        raise ValueError(f"Unsupported JSON payload in {path}")

    rows: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} is not valid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i} must be a JSON object")
        rows.append(row)
    return rows


def _extract_text(row: dict[str, Any], fallback_key: str) -> str:
    for key in OUTPUT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = row.get(fallback_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = _load_json_or_jsonl(path)
    prompts: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = _extract_text(row, "prompt")
        if not prompt:
            raise ValueError(f"{path}: row {idx} has no prompt text")
        raw_id = row.get("id", idx)
        prompts.append({"id": str(raw_id), "prompt": prompt})
    return prompts


def load_outputs(path: Path) -> dict[str, str]:
    rows = _load_json_or_jsonl(path)
    out: dict[str, str] = {}
    for idx, row in enumerate(rows, start=1):
        raw_id = row.get("id")
        if raw_id is None:
            raise ValueError(f"{path}: row {idx} missing 'id'")
        text = _extract_text(row, "output")
        if not text:
            raise ValueError(f"{path}: row {idx} has no output text")
        out[str(raw_id)] = text
    return out


def run_command(template: str, case_id: str, prompt: str, timeout_s: int) -> str:
    mapping = {
        "id": case_id,
        "prompt": prompt,
        "prompt_json": json.dumps(prompt),
    }
    cmd = template.format(**mapping)
    proc = subprocess.run(
        ["zsh", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed for case {case_id} (exit {proc.returncode}):\n"
            f"CMD: {cmd}\nSTDERR:\n{proc.stderr.strip()}"
        )
    output = proc.stdout.strip()
    if not output:
        raise RuntimeError(f"Command produced empty output for case {case_id}: {cmd}")
    return output


def build_cases_from_files(
    prompts_path: Path,
    baseline_path: Path,
    reference_path: Path,
    max_cases: int | None = None,
) -> list[EvalCase]:
    prompts = load_prompts(prompts_path)
    baseline = load_outputs(baseline_path)
    reference = load_outputs(reference_path)

    cases: list[EvalCase] = []
    for p in prompts:
        case_id = p["id"]
        if case_id not in baseline:
            raise ValueError(f"Missing baseline output for id={case_id}")
        if case_id not in reference:
            raise ValueError(f"Missing reference output for id={case_id}")
        cases.append(
            EvalCase(
                case_id=case_id,
                prompt=p["prompt"],
                baseline_output=baseline[case_id],
                reference_output=reference[case_id],
            )
        )

    if max_cases is not None:
        return cases[:max_cases]
    return cases


def build_cases_from_commands(
    prompts_path: Path,
    baseline_cmd: str,
    reference_cmd: str,
    timeout_s: int,
    max_cases: int | None = None,
) -> list[EvalCase]:
    prompts = load_prompts(prompts_path)
    if max_cases is not None:
        prompts = prompts[:max_cases]

    cases: list[EvalCase] = []
    for p in prompts:
        case_id = p["id"]
        prompt = p["prompt"]
        baseline_output = run_command(baseline_cmd, case_id=case_id, prompt=prompt, timeout_s=timeout_s)
        reference_output = run_command(reference_cmd, case_id=case_id, prompt=prompt, timeout_s=timeout_s)
        cases.append(
            EvalCase(
                case_id=case_id,
                prompt=prompt,
                baseline_output=baseline_output,
                reference_output=reference_output,
            )
        )
    return cases


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response")
    return json.loads(match.group(0))


def judge_case_with_anthropic(
    case: EvalCase,
    model: str,
    seed: int,
    client: Any,
) -> dict[str, Any]:
    if random.Random(seed).random() < 0.5:
        a_label = "baseline"
        a_output = case.baseline_output
        b_label = "reference"
        b_output = case.reference_output
    else:
        a_label = "reference"
        a_output = case.reference_output
        b_label = "baseline"
        b_output = case.baseline_output

    system_prompt = (
        "You are a strict evaluator for frontend code generation quality. "
        "Compare Output A and Output B for the same prompt. "
        "Judge based on visual specificity, layout coherence, implementation detail, and non-genericness. "
        "Ignore superficial length differences. Return JSON only."
    )

    user_prompt = f"""
Prompt:
{case.prompt}

Output A:
{a_output}

Output B:
{b_output}

Return JSON only with this schema:
{{
  "winner": "A|B|tie",
  "confidence": 0.0,
  "scores": {{
    "A": {{"visual_specificity": 0-5, "layout_coherence": 0-5, "implementation_detail": 0-5, "non_genericness": 0-5}},
    "B": {{"visual_specificity": 0-5, "layout_coherence": 0-5, "implementation_detail": 0-5, "non_genericness": 0-5}}
  }},
  "rationale": "max 45 words"
}}
""".strip()

    resp = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            text += block.text

    parsed = _extract_json_block(text)
    winner = parsed.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError(f"Invalid winner in judge response: {winner}")

    mapped_winner = "tie"
    if winner == "A":
        mapped_winner = a_label
    elif winner == "B":
        mapped_winner = b_label

    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "winner": mapped_winner,
        "winner_raw": winner,
        "a_label": a_label,
        "b_label": b_label,
        "confidence": parsed.get("confidence"),
        "scores": parsed.get("scores", {}),
        "rationale": parsed.get("rationale", ""),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
    reference_wins = sum(1 for r in results if r["winner"] == "reference")
    ties = sum(1 for r in results if r["winner"] == "tie")
    total = len(results)
    decided = baseline_wins + reference_wins

    return {
        "total_cases": total,
        "decided_cases": decided,
        "baseline_wins": baseline_wins,
        "reference_wins": reference_wins,
        "ties": ties,
        "reference_win_rate_all": round(reference_wins / total, 4) if total else 0.0,
        "reference_win_rate_decided": round(reference_wins / decided, 4) if decided else 0.0,
    }


def render_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Reference Impact Eval Report",
        "",
        "## Summary",
        f"- Total cases: {summary['total_cases']}",
        f"- Decided cases: {summary['decided_cases']}",
        f"- Reference wins: {summary['reference_wins']}",
        f"- Baseline wins: {summary['baseline_wins']}",
        f"- Ties: {summary['ties']}",
        f"- Reference win rate (all): {summary['reference_win_rate_all']:.2%}",
        f"- Reference win rate (decided): {summary['reference_win_rate_decided']:.2%}",
        "",
        "## Per-case Results",
    ]

    for r in results:
        lines.extend(
            [
                f"### Case {r['case_id']}",
                f"- Winner: {r['winner']}",
                f"- Confidence: {r.get('confidence')}",
                f"- Rationale: {r.get('rationale', '').strip()}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs reference-backed generation outputs")
    parser.add_argument("--prompts-file", required=True, help="JSON/JSONL list with {id, prompt}")

    parser.add_argument("--baseline-file", help="JSON/JSONL with {id, output}")
    parser.add_argument("--reference-file", help="JSON/JSONL with {id, output}")

    parser.add_argument("--baseline-cmd", help="Shell template to generate baseline output. Uses {id}, {prompt}, {prompt_json}")
    parser.add_argument("--reference-cmd", help="Shell template to generate reference output. Uses {id}, {prompt}, {prompt_json}")
    parser.add_argument("--timeout-sec", type=int, default=120)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", default="outputs/eval_reference_impact.json")
    parser.add_argument("--output-md", default="outputs/eval_reference_impact.md")
    parser.add_argument("--print-each", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    use_files = bool(args.baseline_file and args.reference_file)
    use_commands = bool(args.baseline_cmd and args.reference_cmd)
    if use_files == use_commands:
        raise ValueError("Provide either --baseline-file/--reference-file or --baseline-cmd/--reference-cmd")

    if use_files:
        cases = build_cases_from_files(
            prompts_path=prompts_path,
            baseline_path=Path(args.baseline_file),
            reference_path=Path(args.reference_file),
            max_cases=args.max_cases,
        )
    else:
        cases = build_cases_from_commands(
            prompts_path=prompts_path,
            baseline_cmd=args.baseline_cmd,
            reference_cmd=args.reference_cmd,
            timeout_s=args.timeout_sec,
            max_cases=args.max_cases,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for judging")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        case_seed = args.seed + idx
        judged = judge_case_with_anthropic(case, model=args.model, seed=case_seed, client=client)
        results.append(judged)
        if args.print_each:
            print(f"[{idx}/{len(cases)}] case={case.case_id} winner={judged['winner']}")

    summary_data = summarize(results)
    payload = {
        "config": {
            "model": args.model,
            "seed": args.seed,
            "max_cases": args.max_cases,
            "prompt_file": str(prompts_path),
            "source_mode": "files" if use_files else "commands",
        },
        "summary": summary_data,
        "results": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary_data, results), encoding="utf-8")

    print(json.dumps(summary_data, indent=2))
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote Markdown report: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
