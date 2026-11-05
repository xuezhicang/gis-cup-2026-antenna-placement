#!/usr/bin/env python3
"""Collect the best completed simulated-annealing island result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def solution_shape(path: Path) -> tuple[int, int]:
    """Return (k, hard score) from one official three-line result."""

    from common.solution_txt import parse_solution_blocks

    blocks = parse_solution_blocks(path)
    if len(blocks) != 1:
        raise ValueError(f"Expected one solution block in {path}")
    return blocks[0].k, len(blocks[0].building_ids)


def collect(output_folder: Path) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for rank_folder in sorted(output_folder.glob("rank_[0-9][0-9][0-9][0-9]")):
        metadata_path = rank_folder / "best.json"
        solution_path = rank_folder / "best_solution.txt"
        if not metadata_path.is_file() or not solution_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            k, hard_score = solution_shape(solution_path)
            recorded_hard = int(metadata["hard_score"])
            soft_score: float | None = float(metadata["soft_score"])
            iteration = int(metadata["iteration"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"Skipping incomplete {rank_folder}: {error}", flush=True)
            continue
        if recorded_hard != hard_score:
            print(
                f"Warning: {rank_folder} metadata says {recorded_hard}, "
                f"but its TXT contains {hard_score} building IDs; using the TXT score.",
                flush=True,
            )
            soft_score = None
        candidates.append(
            {
                "rank_folder": rank_folder,
                "solution_path": solution_path,
                "hard_score": hard_score,
                "soft_score": soft_score,
                "iteration": iteration,
                "k": k,
            }
        )

    if not candidates:
        raise RuntimeError(f"No completed rank result found below {output_folder}")

    winner = max(
        candidates,
        key=lambda item: (
            int(item["hard_score"]),
            (
                float(item["soft_score"])
                if item["soft_score"] is not None
                else float("-inf")
            ),
            -int(str(Path(item["rank_folder"]).name).split("_")[-1]),
        ),
    )
    global_solution = output_folder / "global_best_solution.txt"
    atomic_write_bytes(global_solution, Path(winner["solution_path"]).read_bytes())
    result = {
        "hard_score": int(winner["hard_score"]),
        "soft_score": (
            float(winner["soft_score"])
            if winner["soft_score"] is not None
            else None
        ),
        "k": int(winner["k"]),
        "iteration": int(winner["iteration"]),
        "rank_folder": Path(winner["rank_folder"]).name,
        "source_solution": str(
            Path(winner["solution_path"]).relative_to(output_folder)
        ),
        "solution": global_solution.name,
        "completed_rank_results": len(candidates),
    }
    atomic_write_json(output_folder / "global_best.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    result = collect(args.output_folder)
    soft_display = (
        f"{float(result['soft_score']):.6f}"
        if result["soft_score"] is not None
        else "unknown"
    )
    print(
        "GLOBAL_BEST "
        f"hard={result['hard_score']} soft={soft_display} "
        f"rank={result['rank_folder']} "
        f"solution={args.output_folder / str(result['solution'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
