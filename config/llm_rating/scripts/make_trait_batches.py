"""Create LLM trait-rating batches from the human-rating template."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("config/human_rating/template/trait_foundation_rating_template.tsv")
DEFAULT_OUTPUT_DIR = Path("config/llm_rating/batches")
TRAIT_COLUMNS = ["trait_index", "differential", "left_pole", "right_pole"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TSV batches of trait differentials for LLM rating."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input TSV template. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for batch TSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of traits per batch. Default: 25",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing batch_*.tsv files in the output directory before writing.",
    )
    return parser.parse_args()


def read_traits(input_path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with input_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                missing = [column for column in TRAIT_COLUMNS if column not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Input file is missing required columns: {missing}")
                return [{column: row[column] for column in TRAIT_COLUMNS} for row in reader]
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise RuntimeError(f"Could not read {input_path}")


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_batches(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def main() -> None:
    args = parse_args()
    rows = read_traits(args.input)
    batches = make_batches(rows, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old_batch in args.output_dir.glob("batch_*.tsv"):
            old_batch.unlink()

    manifest_rows = []
    for batch_number, batch_rows in enumerate(batches, start=1):
        batch_name = f"batch_{batch_number:03d}.tsv"
        batch_path = args.output_dir / batch_name
        write_tsv(batch_path, batch_rows, TRAIT_COLUMNS)
        manifest_rows.append(
            {
                "batch_file": batch_name,
                "start_trait_index": batch_rows[0]["trait_index"],
                "end_trait_index": batch_rows[-1]["trait_index"],
                "n_traits": str(len(batch_rows)),
            }
        )

    write_tsv(
        args.output_dir / "batch_manifest.tsv",
        manifest_rows,
        ["batch_file", "start_trait_index", "end_trait_index", "n_traits"],
    )
    print(f"Wrote {len(batches)} batches to {args.output_dir}")


if __name__ == "__main__":
    main()
