"""Rate trait batches with Gemini or DeepSeek APIs.

API keys are read from environment variables:
- GEMINI_API_KEY for --provider gemini
- DEEPSEEK_API_KEY for --provider deepseek
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LLM_DIR = ROOT / "config" / "llm_rating"
HUMAN_TEMPLATE_DIR = ROOT / "config" / "human_rating" / "template"

DEFAULT_PROMPT = LLM_DIR / "prompts" / "api_batch_trait_rating_prompt.md"
DEFAULT_BATCH_DIR = LLM_DIR / "batches"
DEFAULT_OUTPUT_DIR = LLM_DIR / "outputs"
DEFAULT_RAW_DIR = LLM_DIR / "raw_responses"
DEFAULT_DEFINITIONS = HUMAN_TEMPLATE_DIR / "foundation_definition.txt"
DEFAULT_DICTIONARY = HUMAN_TEMPLATE_DIR / "foundation_dictionary.txt"

RATING_COLUMNS = ["care", "fairness", "loyalty", "authority", "purity", "liberty", "general"]
OUTPUT_COLUMNS = [
    "trait_index",
    "differential",
    "left_pole",
    "right_pole",
    *RATING_COLUMNS,
    "notes",
]
INPUT_COLUMNS = ["trait_index", "differential", "left_pole", "right_pole"]

GEMINI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "trait_index": {"type": "integer"},
                    "differential": {"type": "string"},
                    "left_pole": {"type": "string"},
                    "right_pole": {"type": "string"},
                    "care": {"type": "integer", "enum": [0, 1]},
                    "fairness": {"type": "integer", "enum": [0, 1]},
                    "loyalty": {"type": "integer", "enum": [0, 1]},
                    "authority": {"type": "integer", "enum": [0, 1]},
                    "purity": {"type": "integer", "enum": [0, 1]},
                    "liberty": {"type": "integer", "enum": [0, 1]},
                    "general": {"type": "integer", "enum": [0, 1]},
                    "notes": {"type": "string"},
                },
                "required": OUTPUT_COLUMNS,
                "propertyOrdering": OUTPUT_COLUMNS,
            },
        }
    },
    "required": ["ratings"],
    "propertyOrdering": ["ratings"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM trait ratings over TSV batches.")
    parser.add_argument("--provider", choices=["gemini", "deepseek"], required=True)
    parser.add_argument("--model", help="Model name. Defaults depend on provider.")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--batch", type=Path, help="Run one specific batch TSV instead of all batches.")
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Limit number of batches for a test run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing batch outputs.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in INPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return [{column: row[column] for column in INPUT_COLUMNS} for row in reader]


def rows_to_tsv(rows: list[dict[str, str]]) -> str:
    lines = ["\t".join(INPUT_COLUMNS)]
    for row in rows:
        lines.append("\t".join(row[column] for column in INPUT_COLUMNS))
    return "\n".join(lines)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(args: argparse.Namespace, batch_rows: list[dict[str, str]]) -> str:
    template = read_text(args.prompt)
    return (
        template.replace("{foundation_definitions}", read_text(args.definitions).strip())
        .replace("{foundation_dictionary}", read_text(args.dictionary).strip())
        .replace("{batch_tsv}", rows_to_tsv(batch_rows))
    )


def request_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    max_retries = 5
    base_delay = 4.0
    for attempt in range (max_retries): 
        
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
        #handle 503 error
            if error.code == 503 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Server busy. Retry in {delay}s...")
                time.sleep(delay)
                continue
            #handle bad request or invalid key
            raise RuntimeError(f"HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Timeout error. Retry in {delay}s...")
                time.sleep(delay)
                continue 
            raise error

def call_gemini(prompt: str, model: str, api_key: str, temperature: float, max_tokens: int) -> tuple[dict[str, Any], str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": GEMINI_SCHEMA,
        },
    }
    raw = request_json(
        url,
        {"Content-Type": "application/json", "x-goog-api-key": api_key},
        payload,
    )
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    return raw, text


def call_deepseek(prompt: str, model: str, api_key: str, temperature: float, max_tokens: int) -> tuple[dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You return strict JSON only. Do not wrap JSON in Markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    raw = request_json(
        "https://api.deepseek.com/chat/completions",
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        payload,
    )
    text = raw["choices"][0]["message"]["content"]
    return raw, text


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_rating(value: Any) -> int:
    if value in (1, "+1", "1"):
        return 1
    if value in (0, "0"):
        return 0
    raise ValueError(f"Invalid rating value: {value!r}")


def validate_ratings(parsed: dict[str, Any], batch_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ratings = parsed.get("ratings")
    if not isinstance(ratings, list):
        raise ValueError("Response JSON must contain a 'ratings' list.")
    if len(ratings) != len(batch_rows):
        raise ValueError(f"Expected {len(batch_rows)} ratings, got {len(ratings)}.")

    expected_by_index = {str(row["trait_index"]): row for row in batch_rows}
    cleaned_rows = []
    seen_indices = set()

    for rating in ratings:
        trait_index = str(rating.get("trait_index", "")).strip()
        if trait_index not in expected_by_index:
            raise ValueError(f"Unexpected trait_index in response: {trait_index!r}")
        if trait_index in seen_indices:
            raise ValueError(f"Duplicate trait_index in response: {trait_index}")
        seen_indices.add(trait_index)

        source = expected_by_index[trait_index]
        cleaned = {
            "trait_index": source["trait_index"],
            "differential": source["differential"],
            "left_pole": source["left_pole"],
            "right_pole": source["right_pole"],
        }
        for column in RATING_COLUMNS:
            cleaned[column] = normalize_rating(rating.get(column))
        cleaned["notes"] = str(rating.get("notes", "")).replace("\t", " ").replace("\n", " ").strip()
        cleaned_rows.append(cleaned)

    return sorted(cleaned_rows, key=lambda row: int(row["trait_index"]))


def get_batches(args: argparse.Namespace) -> list[Path]:
    if args.batch:
        return [args.batch]
    batches = sorted(args.batch_dir.glob("batch_*.tsv"))
    batches = [path for path in batches if re.fullmatch(r"batch_\d{3}\.tsv", path.name)]
    if args.limit:
        batches = batches[: args.limit]
    return batches


def provider_defaults(provider: str) -> tuple[str, str]:
    if provider == "gemini":
        return "gemini-2.5-flash", "GEMINI_API_KEY"
    if provider == "deepseek":
        return "deepseek-v4-flash", "DEEPSEEK_API_KEY"
    raise ValueError(f"Unknown provider: {provider}")


def run_batch(args: argparse.Namespace, batch_path: Path, model: str, api_key: str) -> None:
    provider_dir = f"{args.provider}_{model}"
    output_path = args.output_dir / provider_dir / batch_path.name
    raw_path = args.raw_dir / provider_dir / f"{batch_path.stem}.json"

    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {output_path}")
        return

    batch_rows = read_tsv(batch_path)
    prompt = build_prompt(args, batch_rows)

    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 2):
        try:
            if args.provider == "gemini":
                raw, text = call_gemini(prompt, model, api_key, args.temperature, args.max_output_tokens)
            else:
                raw, text = call_deepseek(prompt, model, api_key, args.temperature, args.max_output_tokens)

            parsed = extract_json(text)
            cleaned_rows = validate_ratings(parsed, batch_rows)

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps({"api_response": raw, "parsed": parsed}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            write_tsv(output_path, cleaned_rows)
            print(f"Wrote {output_path}")
            return
        except Exception as error:
            last_error = error
            if attempt <= args.max_retries:
                wait_seconds = args.sleep_seconds * attempt
                print(f"Attempt {attempt} failed for {batch_path.name}: {error}. Retrying in {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Failed {batch_path.name} after retries: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    default_model, env_name = provider_defaults(args.provider)
    model = args.model or default_model
    api_key = os.environ.get(env_name)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {env_name} before running this script.")

    batches = get_batches(args)
    if not batches:
        raise SystemExit(f"No batch files found in {args.batch_dir}")

    for batch_path in batches:
        run_batch(args, batch_path, model, api_key)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
