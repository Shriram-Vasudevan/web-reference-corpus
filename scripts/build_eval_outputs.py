#!/usr/bin/env python3
"""Generate baseline/reference output JSONL files for eval runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _load_prompts(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
            rows = payload["items"]
        else:
            raise ValueError(f"Unsupported prompt JSON format in {path}")
    else:
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} invalid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i} must be a JSON object")
            rows.append(obj)

    prompts: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        case_id = str(row.get("id", i))
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: row {i} missing prompt text")
        prompts.append({"id": case_id, "prompt": prompt.strip()})
    return prompts


def _run_command(cmd_template: str, case_id: str, prompt: str, timeout_sec: int) -> str:
    cmd = cmd_template.format(id=case_id, prompt=prompt, prompt_json=json.dumps(prompt))
    proc = subprocess.run(
        ["zsh", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed for case={case_id}, exit={proc.returncode}\n"
            f"CMD: {cmd}\nSTDERR:\n{proc.stderr.strip()}"
        )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"Command returned empty stdout for case={case_id}\nCMD: {cmd}")
    return out


def _extract_output(stdout: str, parse_mode: str, json_key: str | None) -> str:
    if parse_mode == "raw":
        return stdout.strip()

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON stdout but parse failed: {exc}") from exc

    if json_key:
        value: Any = payload
        for part in json_key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"JSON key '{json_key}' not found in command output")
            value = value[part]
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=True)
        return value.strip()

    if isinstance(payload, str):
        return payload.strip()
    return json.dumps(payload, ensure_ascii=True)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eval output files by running generation commands")
    parser.add_argument("--prompts-file", required=True, help="JSON/JSONL with {id,prompt}")
    parser.add_argument("--baseline-cmd", required=True, help="Command template for baseline generation")
    parser.add_argument("--reference-cmd", required=True, help="Command template for reference-backed generation")
    parser.add_argument("--baseline-out", default="evals/baseline_outputs.generated.jsonl")
    parser.add_argument("--reference-out", default="evals/reference_outputs.generated.jsonl")
    parser.add_argument("--baseline-parse-mode", choices=["raw", "json"], default="raw")
    parser.add_argument("--reference-parse-mode", choices=["raw", "json"], default="raw")
    parser.add_argument("--baseline-json-key", default=None, help="Dot path for output field when parse-mode=json")
    parser.add_argument("--reference-json-key", default=None, help="Dot path for output field when parse-mode=json")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--print-each", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = _load_prompts(Path(args.prompts_file))
    if args.max_cases is not None:
        prompts = prompts[: args.max_cases]

    baseline_rows: list[dict[str, str]] = []
    reference_rows: list[dict[str, str]] = []

    total = len(prompts)
    for idx, item in enumerate(prompts, start=1):
        case_id = item["id"]
        prompt = item["prompt"]

        baseline_stdout = _run_command(args.baseline_cmd, case_id=case_id, prompt=prompt, timeout_sec=args.timeout_sec)
        reference_stdout = _run_command(args.reference_cmd, case_id=case_id, prompt=prompt, timeout_sec=args.timeout_sec)

        baseline_output = _extract_output(
            baseline_stdout,
            parse_mode=args.baseline_parse_mode,
            json_key=args.baseline_json_key,
        )
        reference_output = _extract_output(
            reference_stdout,
            parse_mode=args.reference_parse_mode,
            json_key=args.reference_json_key,
        )

        baseline_rows.append({"id": case_id, "output": baseline_output})
        reference_rows.append({"id": case_id, "output": reference_output})

        if args.print_each:
            print(f"[{idx}/{total}] wrote outputs for case={case_id}")

    baseline_out = Path(args.baseline_out)
    reference_out = Path(args.reference_out)
    _write_jsonl(baseline_out, baseline_rows)
    _write_jsonl(reference_out, reference_rows)

    print(f"Wrote baseline outputs: {baseline_out}")
    print(f"Wrote reference outputs: {reference_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
