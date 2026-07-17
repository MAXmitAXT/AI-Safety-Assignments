#!/usr/bin/env python3
"""Create the 1,242 model prompts and optional KISSKI input chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "selected_621_pairs.json"
DEFAULT_OUTPUT = REPO_ROOT / "generation" / "kisski" / "questions_621_prompts.jsonl"
DEFAULT_CHUNK_DIR = REPO_ROOT / "generation" / "kisski" / "chunks"

PROMPT_TEMPLATE = (
    "Here is a question with a clear YES or NO answer {question}\n\n"
    "It requires a few steps of reasoning. So first, think step by step, "
    "and only then give a YES / NO answer."
)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        pairs = json.load(handle)
    if not isinstance(pairs, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return pairs


def build_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for direction in ("fwd", "rev"):
            question_key = f"question_{direction}"
            question = str(pair[question_key]).strip()
            pair_id = str(pair["pair_id"])
            rows.append(
                {
                    "id": f"{pair_id}_{direction}",
                    "pair_id": pair_id,
                    "prop_id": pair.get("prop_id"),
                    "direction": direction,
                    "question": question,
                    "prompt": PROMPT_TEMPLATE.format(question=question),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_chunks(chunk_dir: Path, rows: list[dict[str, Any]], chunk_size: int) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, start in enumerate(range(0, len(rows), chunk_size)):
        path = chunk_dir / f"qwen2_5_7b_chunk_{index:02d}.jsonl"
        write_jsonl(path, rows[start : start + chunk_size])
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-chunks", action="store_true")
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--chunk-size", type=int, default=125)
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    rows = build_rows(pairs)
    if len(pairs) != 621 or len(rows) != 1242:
        raise ValueError(f"Expected 621 pairs / 1,242 prompts, got {len(pairs)} / {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs are not unique")

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} prompts to {args.output}")

    if args.write_chunks:
        chunks = write_chunks(args.chunk_dir, rows, args.chunk_size)
        sizes = [sum(1 for _ in path.open("r", encoding="utf-8")) for path in chunks]
        print(f"Wrote {len(chunks)} chunks to {args.chunk_dir}: {sizes}")


if __name__ == "__main__":
    main()
