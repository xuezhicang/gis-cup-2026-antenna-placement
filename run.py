#!/usr/bin/env python3
"""Run the whole antenna-placement pipeline from one config file.

    python run.py                       # reads config.toml
    python run.py --config my.toml
    python run.py --steps 3,4           # rerun only some steps; earlier outputs must exist

Step 1 (visibility) depends only on the GeoJSON and the step-1 settings, so its
result is cached in <output_dir>/cache and reused when tau, k or later-step
settings change.  Steps 2-4 run once per (tau, k) sub-problem.  The final
answer for every sub-problem goes into <output_dir>/submission.txt, and
<output_dir>/scores.tsv lists the hard score reached after each step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        sys.exit("Python 3.10 needs the 'tomli' package to read config.toml: pip install tomli")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.solution_txt import parse_solution_blocks  # noqa: E402
from step4_annealing.select_seeds import (  # noqa: E402
    deduplicate,
    load_candidate,
    report_rows,
    select_seeds,
)


REQUIRED = object()

# Every key a user may set, with its default.  Keys in the step sections map
# one-to-one to the command-line options of that step's script
# (for example ``edge_spacing`` -> ``--edge-spacing``).
DEFAULTS: dict[str, dict[str, object]] = {
    "problem": {
        "geojson": REQUIRED,
        "tau": REQUIRED,
        "k": REQUIRED,
        "building_id_column": "id",
    },
    "run": {
        "output_dir": "outputs",
        "workers": 1,
        "random_seed": 42,
        "mpi_launcher": "mpiexec",
        "mpi_args": [],
    },
    "step1_visibility": {
        "edge_spacing": 20.0,
        "max_points": 0,
        "max_leaf_size": 64,
        "ring_step": 19.0,
    },
    "step2_greedy": {
        "method": "two_step",
        "alpha": 0.8,
        "lookahead_width": 3,
        "grid_size_meters": 200.0,
        "grid_top_k": 5,
        "global_top_fraction": 0.0,
        "random_count": 0,
    },
    "step3_point_swap": {
        "enabled": True,
        "solutions": 1,
        "beam_width": 1,
        "max_generations": 10,
        "max_seconds": 300.0,
    },
    "step4_annealing": {
        "enabled": True,
        "max_seconds": 600.0,
        "seeds": 1,
        "seed_score_tolerance": 0.05,
        "advanced": {},
    },
}

# Optional simulated-annealing settings accepted in [step4_annealing.advanced].
SA_ADVANCED_KEYS = {
    "candidate_pool_size": int,
    "top_candidates": int,
    "proposal_rank_temperature": float,
    "random_candidate_probability": float,
    "one_swap_probability": float,
    "two_swap_probability": float,
    "ruin_probability": float,
    "ruin_size": int,
    "hard_loss_acceptance_start": float,
    "hard_loss_acceptance_end": float,
    "soft_worse_acceptance_start": float,
    "soft_worse_acceptance_end": float,
    "temperature_log_spread": float,
    "cycle_iterations": int,
    "stagnation_iterations": int,
    "calibration_trials": int,
}

GREEDY_METHODS = ("two_step", "multistart")
STEP_NAMES = {
    1: "step1_visibility",
    2: "step2_greedy",
    3: "step3_point_swap",
    4: "step4_annealing",
}


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise PipelineError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    unknown_sections = sorted(set(raw) - set(DEFAULTS))
    if unknown_sections:
        raise PipelineError(
            f"Unknown section(s) in {path}: {', '.join(unknown_sections)}. "
            f"Valid sections: {', '.join(DEFAULTS)}"
        )

    config: dict[str, dict[str, object]] = {}
    for section, defaults in DEFAULTS.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise PipelineError(f"[{section}] must be a table")
        unknown = sorted(set(values) - set(defaults))
        if unknown:
            raise PipelineError(
                f"Unknown key(s) in [{section}]: {', '.join(unknown)}. "
                f"Valid keys: {', '.join(defaults)}"
            )
        merged = {}
        for key, default in defaults.items():
            if key in values:
                merged[key] = values[key]
            elif default is REQUIRED:
                raise PipelineError(f"Missing required key '{key}' in [{section}]")
            else:
                merged[key] = default
        config[section] = merged

    validate_config(config, path.parent)
    return config


def _number(section: str, key: str, value: object, *, integer: bool = False,
            minimum: float | None = None, exclusive: bool = False) -> float | int:
    valid_type = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, valid_type):
        kind = "an integer" if integer else "a number"
        raise PipelineError(f"[{section}] {key} must be {kind}, got {value!r}")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        relation = ">" if exclusive else ">="
        raise PipelineError(f"[{section}] {key} must be {relation} {minimum}, got {value!r}")
    return value


def _number_list(section: str, key: str, value: object, *, integer: bool = False,
                 minimum: float | None = None) -> list:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise PipelineError(f"[{section}] {key} must not be an empty list")
    for item in values:
        _number(section, key, item, integer=integer, minimum=minimum)
    if len(set(values)) != len(values):
        raise PipelineError(f"[{section}] {key} values must not repeat")
    return list(values)


def validate_config(config: dict[str, dict[str, object]], base_dir: Path) -> None:
    problem = config["problem"]
    geojson = Path(str(problem["geojson"]))
    if not geojson.is_absolute():
        geojson = (base_dir / geojson).resolve()
    if not geojson.is_file():
        raise PipelineError(
            f"GeoJSON not found: {geojson}\n"
            "Download the data first with: bash data/download_data.sh"
        )
    problem["geojson"] = geojson

    tau_values = problem["tau"] if isinstance(problem["tau"], list) else [problem["tau"]]
    k_values = problem["k"] if isinstance(problem["k"], list) else [problem["k"]]
    if not tau_values or not k_values:
        raise PipelineError("[problem] tau and k must not be empty")
    for tau in tau_values:
        _number("problem", "tau", tau)
        if not 0.0 < float(tau) <= 1.0:
            raise PipelineError(f"[problem] every tau must be in (0, 1], got {tau!r}")
    for k in k_values:
        _number("problem", "k", k, integer=True, minimum=1)
    if len(set(map(float, tau_values))) != len(tau_values) or len(set(k_values)) != len(k_values):
        raise PipelineError("[problem] tau and k values must not repeat")
    problem["tau"] = sorted(float(tau) for tau in tau_values)
    problem["k"] = sorted(int(k) for k in k_values)

    run = config["run"]
    output_dir = Path(str(run["output_dir"]))
    run["output_dir"] = output_dir if output_dir.is_absolute() else (base_dir / output_dir).resolve()
    _number("run", "workers", run["workers"], integer=True, minimum=1)
    _number("run", "random_seed", run["random_seed"], integer=True, minimum=0)
    if not isinstance(run["mpi_args"], list) or not all(isinstance(a, str) for a in run["mpi_args"]):
        raise PipelineError("[run] mpi_args must be a list of strings")

    visibility = config["step1_visibility"]
    _number("step1_visibility", "edge_spacing", visibility["edge_spacing"], minimum=0, exclusive=True)
    _number("step1_visibility", "max_points", visibility["max_points"], integer=True, minimum=0)
    _number("step1_visibility", "max_leaf_size", visibility["max_leaf_size"], integer=True, minimum=1)
    _number("step1_visibility", "ring_step", visibility["ring_step"], minimum=0, exclusive=True)

    greedy = config["step2_greedy"]
    if greedy["method"] not in GREEDY_METHODS:
        raise PipelineError(
            f"[step2_greedy] method must be one of {', '.join(GREEDY_METHODS)}, "
            f"got {greedy['method']!r}"
        )
    # alpha and lookahead_width may be one value or a list; every combination
    # gives one greedy solution per sub-problem.
    greedy["alpha"] = _number_list("step2_greedy", "alpha", greedy["alpha"])
    for alpha in greedy["alpha"]:
        if not 0.0 <= alpha <= 1.0:
            raise PipelineError("[step2_greedy] every alpha must be in [0, 1]")
    greedy["lookahead_width"] = _number_list(
        "step2_greedy", "lookahead_width", greedy["lookahead_width"], integer=True, minimum=1
    )
    _number("step2_greedy", "grid_size_meters", greedy["grid_size_meters"], minimum=0, exclusive=True)
    _number("step2_greedy", "grid_top_k", greedy["grid_top_k"], integer=True, minimum=0)
    _number("step2_greedy", "global_top_fraction", greedy["global_top_fraction"], minimum=0)
    _number("step2_greedy", "random_count", greedy["random_count"], integer=True, minimum=0)

    swap = config["step3_point_swap"]
    if not isinstance(swap["enabled"], bool):
        raise PipelineError("[step3_point_swap] enabled must be true or false")
    _number("step3_point_swap", "solutions", swap["solutions"], integer=True, minimum=1)
    _number("step3_point_swap", "beam_width", swap["beam_width"], integer=True, minimum=1)
    _number("step3_point_swap", "max_generations", swap["max_generations"], integer=True, minimum=0)
    _number("step3_point_swap", "max_seconds", swap["max_seconds"], minimum=0)

    annealing = config["step4_annealing"]
    if not isinstance(annealing["enabled"], bool):
        raise PipelineError("[step4_annealing] enabled must be true or false")
    if annealing["enabled"]:
        _number("step4_annealing", "max_seconds", annealing["max_seconds"], minimum=0, exclusive=True)
    seeds = _number("step4_annealing", "seeds", annealing["seeds"], integer=True, minimum=1)
    workers = int(run["workers"])
    if seeds > workers or workers % seeds:
        allowed = ", ".join(str(n) for n in range(1, workers + 1) if workers % n == 0)
        raise PipelineError(
            f"[step4_annealing] seeds = {seeds} does not fit workers = {workers}: every seed "
            f"needs the same number of processes, so seeds must divide workers "
            f"(allowed: {allowed})"
        )
    tolerance = _number("step4_annealing", "seed_score_tolerance", annealing["seed_score_tolerance"], minimum=0)
    if tolerance >= 1:
        raise PipelineError("[step4_annealing] seed_score_tolerance must be below 1")
    advanced = annealing["advanced"]
    if not isinstance(advanced, dict):
        raise PipelineError("[step4_annealing.advanced] must be a table")
    unknown = sorted(set(advanced) - set(SA_ADVANCED_KEYS))
    if unknown:
        raise PipelineError(
            f"Unknown key(s) in [step4_annealing.advanced]: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(SA_ADVANCED_KEYS)}"
        )
    for key, kind in SA_ADVANCED_KEYS.items():
        if key in advanced:
            _number("step4_annealing.advanced", key, advanced[key], integer=kind is int)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def subproblem_name(tau: float, k: int) -> str:
    return f"tau_{tau:g}_k_{k}".replace(".", "p")


def two_step_dir_token(tau: float) -> str:
    # Mirrors filename_number() in step2_greedy/run_two_step_greedy.py.
    return f"{tau:g}".replace("-", "m").replace(".", "p")


def multistart_dir_token(tau: float) -> str:
    # Mirrors tau_token() in step2_greedy/multistart_greedy.py.
    return repr(float(tau)).replace("-", "m").replace("+", "").replace(".", "p")


def hard_score(path: Path) -> int:
    blocks = parse_solution_blocks(path)
    if len(blocks) != 1:
        raise PipelineError(f"Expected one solution block in {path}")
    return len(blocks[0].building_ids)


def best_first(paths: list[Path]) -> list[Path]:
    """Sort solution files by hard score, best first (ties by name)."""

    return sorted(paths, key=lambda path: (-hard_score(path), str(path)))


def distinct_best_first(paths: list[Path]) -> list[Path]:
    """Best-first solution files, keeping one file per distinct antenna set."""

    candidates = [load_candidate(path, str(path)) for path in paths]
    return [candidate.path for candidate in deduplicate(candidates)]


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


class Runner:
    """Starts step scripts, with or without MPI, and mirrors their output to a log."""

    def __init__(self, run_config: dict[str, object]) -> None:
        self.workers = int(run_config["workers"])
        self.launcher = str(run_config["mpi_launcher"])
        self.mpi_args = list(run_config["mpi_args"])
        if self.workers > 1:
            if shutil.which(self.launcher) is None:
                raise PipelineError(
                    f"workers = {self.workers} needs an MPI launcher, but "
                    f"'{self.launcher}' was not found. Install MPI (e.g. OpenMPI) "
                    "or set workers = 1 in config.toml."
                )
            if importlib.util.find_spec("mpi4py") is None:
                raise PipelineError(
                    f"workers = {self.workers} needs the mpi4py package: "
                    "pip install mpi4py (or set workers = 1)."
                )

    def run(self, script: str, args: list[object], log_path: Path, *, processes: int | None = None) -> None:
        processes = self.workers if processes is None else max(1, min(processes, self.workers))
        command = [sys.executable, "-u", str(PROJECT_ROOT / script), *map(str, args)]
        if processes > 1:
            command = [self.launcher, *self.mpi_args, "-np", str(processes), *command]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            print(f"  $ {shlex.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                sys.stdout.write("    " + line)
            return_code = process.wait()
        if return_code != 0:
            raise PipelineError(f"{script} failed with exit code {return_code}; see {log_path}")


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def visibility_cache(config: dict) -> tuple[Path, dict[str, object]]:
    geojson: Path = config["problem"]["geojson"]
    settings = config["step1_visibility"]
    stat = geojson.stat()
    fingerprint = {
        "geojson": str(geojson),
        "geojson_bytes": stat.st_size,
        "geojson_mtime_ns": stat.st_mtime_ns,
        "edge_spacing": float(settings["edge_spacing"]),
        "max_points": int(settings["max_points"]),
        "max_leaf_size": int(settings["max_leaf_size"]),
        "ring_step": float(settings["ring_step"]),
    }
    name = f"visibility_{geojson.stem}_{float(settings['edge_spacing']):g}m"
    if settings["max_points"]:
        name += f"_first{settings['max_points']}"
    name = name.replace(".", "p")
    return config["run"]["output_dir"] / "cache" / name, fingerprint


def step1_visibility(config: dict, runner: Runner) -> tuple[Path, Path]:
    cache_dir, fingerprint = visibility_cache(config)
    segments = cache_dir / "visible_segments.csv"
    candidates = cache_dir / "candidate_points.gpkg"
    marker = cache_dir / "done.json"
    if marker.is_file() and segments.is_file() and candidates.is_file():
        if json.loads(marker.read_text()) == fingerprint:
            print(f"  Reusing cached visibility table: {cache_dir}")
            return segments, candidates
    settings = config["step1_visibility"]
    fresh_dir(cache_dir)
    limit = ["--max-points", settings["max_points"]] if settings["max_points"] else []
    runner.run(
        "step1_visibility/compute_visible_segments.py",
        [
            "--geojson", config["problem"]["geojson"],
            "--points-from-edges",
            "--edge-spacing", settings["edge_spacing"],
            *limit,
            "--max-leaf-size", settings["max_leaf_size"],
            "--ring-step", settings["ring_step"],
            "--output-table-file", segments,
            "--candidate-points-gpkg", candidates,
        ],
        cache_dir / "log.txt",
    )
    marker.write_text(json.dumps(fingerprint, indent=2) + "\n")
    return segments, candidates


def require_visibility(config: dict) -> tuple[Path, Path]:
    cache_dir, fingerprint = visibility_cache(config)
    marker = cache_dir / "done.json"
    if not marker.is_file() or json.loads(marker.read_text()) != fingerprint:
        raise PipelineError("Step 1 output is missing or out of date; include step 1 in --steps")
    return cache_dir / "visible_segments.csv", cache_dir / "candidate_points.gpkg"


Results = dict[tuple[float, int], list[Path]]   # (tau, k) -> solution files, best first


def step2_greedy(config: dict, runner: Runner, segments: Path, candidates: Path) -> Results:
    output_dir = fresh_dir(config["run"]["output_dir"] / "step2_greedy")
    problem = config["problem"]
    greedy = config["step2_greedy"]
    found: dict[tuple[float, int], list[tuple[str, Path]]] = {
        (tau, k): [] for tau in problem["tau"] for k in problem["k"]
    }

    if greedy["method"] == "two_step":
        # One call runs every (tau, k, alpha, lookahead_width) combination;
        # with MPI the combinations are shared between the processes.
        combinations = len(greedy["alpha"]) * len(greedy["lookahead_width"])
        runner.run(
            "step2_greedy/run_two_step_greedy.py",
            [
                "--segments-csv", segments,
                "--candidate-points-gpkg", candidates,
                "--geojson", problem["geojson"],
                "--building-id-column", problem["building_id_column"],
                "--tau", *problem["tau"],
                "--antenna-count", *problem["k"],
                "--alpha-values", *greedy["alpha"],
                "--lookahead-widths", *greedy["lookahead_width"],
                "--output-root", output_dir,
            ],
            output_dir / "log.txt",
            processes=len(problem["tau"]) * len(problem["k"]) * combinations,
        )
        for (tau, k), items in found.items():
            folder = output_dir / f"tau_{two_step_dir_token(tau)}_k_{k}"
            for path in sorted(folder.glob("*.txt")):
                # lookahead_3_alpha_0p8_score_166.txt -> lookahead_3_alpha_0p8
                items.append((path.stem.split("_score_")[0], path))
    else:
        for k in problem["k"]:
            for alpha in greedy["alpha"]:
                run_dir = output_dir / f"multistart_k_{k}_alpha_{number_token(alpha)}"
                runner.run(
                    "step2_greedy/multistart_greedy.py",
                    [
                        "--segments-csv", segments,
                        "--candidate-points-gpkg", candidates,
                        "--geojson", problem["geojson"],
                        "--building-id-column", problem["building_id_column"],
                        "--tau", *problem["tau"],
                        "--antenna-count", k,
                        "--alpha", alpha,
                        "--grid-size-meters", greedy["grid_size_meters"],
                        "--grid-top-k", greedy["grid_top_k"],
                        "--global-top-fraction", greedy["global_top_fraction"],
                        "--random-count", greedy["random_count"],
                        "--random-seed", config["run"]["random_seed"],
                        "--output-dir", run_dir,
                    ],
                    output_dir / "log.txt",
                )
                for tau in problem["tau"]:
                    setting = run_dir / f"tau_{multistart_dir_token(tau)}_k_{k}"
                    for path in sorted((setting / "top_solutions").glob("rank_*.txt")):
                        items = found[(tau, k)]
                        items.append((f"multistart_alpha_{number_token(alpha)}_{path.stem}", path))

    # Copy every solution to candidates/<sub-problem>/ and the best to best/.
    results: Results = {}
    for (tau, k), items in found.items():
        if not items:
            raise PipelineError(f"Step 2 produced no solution for tau={tau:g}, k={k}")
        folder = output_dir / "candidates" / subproblem_name(tau, k)
        folder.mkdir(parents=True)
        copies = []
        for name, path in items:
            target = folder / f"{name}.txt"
            shutil.copyfile(path, target)
            copies.append(target)
        results[(tau, k)] = best_first(copies)
        best_dir = output_dir / "best"
        best_dir.mkdir(exist_ok=True)
        shutil.copyfile(results[(tau, k)][0], best_dir / f"{subproblem_name(tau, k)}.txt")
        print(f"  tau={tau:g} k={k}: {len(copies)} greedy solution(s), "
              f"best hard score {hard_score(results[(tau, k)][0])}")
    return results


def step3_point_swap(config: dict, runner: Runner, segments: Path, inputs: Results) -> Results:
    output_dir = fresh_dir(config["run"]["output_dir"] / "step3_point_swap")
    swap = config["step3_point_swap"]
    results: Results = {}
    for (tau, k), paths in sorted(inputs.items()):
        name = subproblem_name(tau, k)
        chosen = distinct_best_first(paths)[: int(swap["solutions"])]
        swapped = []
        for index, input_path in enumerate(chosen, start=1):
            folder = output_dir / name / input_path.stem
            print(f"\n  -- point swap for tau={tau:g}, k={k}: solution {index}/{len(chosen)} "
                  f"({input_path.stem}, hard score {hard_score(input_path)})")
            runner.run(
                "step3_point_swap/beam_point_swap.py",
                [
                    "--input-result", input_path,
                    "--segments-csv", segments,
                    "--geojson", config["problem"]["geojson"],
                    "--building-id-column", config["problem"]["building_id_column"],
                    "--beam-width", swap["beam_width"],
                    "--max-generations", swap["max_generations"],
                    "--max-seconds", swap["max_seconds"],
                    "--output-folder", folder,
                ],
                output_dir / f"{name}.log",
                processes=k * int(swap["beam_width"]),
            )
            swapped.append(folder / "best_solution.txt")
        results[(tau, k)] = best_first(swapped)
        best_dir = output_dir / "best"
        best_dir.mkdir(exist_ok=True)
        shutil.copyfile(results[(tau, k)][0], best_dir / f"{name}.txt")
    return results


def largest_divisor_at_most(number: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if number % d == 0)


def step4_annealing(config: dict, runner: Runner, segments: Path, inputs: Results) -> Results:
    output_dir = fresh_dir(config["run"]["output_dir"] / "step4_annealing")
    annealing = config["step4_annealing"]
    advanced = [
        item
        for key, value in annealing["advanced"].items()
        for item in (f"--{key.replace('_', '-')}", value)
    ]
    results: Results = {}
    for (tau, k), paths in sorted(inputs.items()):
        name = subproblem_name(tau, k)
        folder = output_dir / name
        log_path = output_dir / f"{name}.log"

        # Pick the seeds with the quality + building-diversity rule.
        selection = select_seeds(
            [load_candidate(path, path.parent.name if path.name == "best_solution.txt" else path.stem)
             for path in paths],
            int(annealing["seeds"]),
            annealing["seed_score_tolerance"],
        )
        seeds = selection.seeds
        # Every seed needs the same number of processes.  Fewer distinct
        # solutions than requested seeds can break that, so trim if needed.
        usable = largest_divisor_at_most(runner.workers, len(seeds))
        seeds = seeds[:usable]
        if len(seeds) < int(annealing["seeds"]):
            print(f"  Note: tau={tau:g}, k={k} has {selection.distinct_candidates} distinct "
                  f"solution(s); using {len(seeds)} seed(s) instead of {annealing['seeds']}.")
        (output_dir / f"{name}_seeds.tsv").write_text(
            "\n".join(report_rows(selection)) + "\n", encoding="utf-8"
        )
        levels = runner.workers // len(seeds)
        print(f"\n  -- simulated annealing for tau={tau:g}, k={k} "
              f"({format_seconds(float(annealing['max_seconds']))}): "
              f"{len(seeds)} seed(s) x {levels} temperature(s), from "
              f"{selection.distinct_candidates} distinct solution(s)")
        for seed in seeds:
            print(f"     seed {seed.rank}: hard score {seed.candidate.score}, "
                  f"building diversity {float(seed.building_diversity):.3f}  ({seed.candidate.name})")
        seed_args = [item for seed in seeds for item in ("--input-result", seed.candidate.path)]
        runner.run(
            "step4_annealing/simulated_annealing.py",
            [
                *seed_args,
                "--segments-csv", segments,
                "--geojson", config["problem"]["geojson"],
                "--building-id-column", config["problem"]["building_id_column"],
                "--output-folder", folder,
                "--max-seconds", annealing["max_seconds"],
                "--random-seed-base", config["run"]["random_seed"],
                # Processes = seeds x temperature profiles.
                "--temperature-profile-levels", levels,
                *advanced,
            ],
            log_path,
        )
        runner.run("step4_annealing/collect_results.py", [folder], log_path, processes=1)
        results[(tau, k)] = [folder / "global_best_solution.txt"]
    return results


def write_summary(config: dict, stage_results: dict[str, Results]) -> None:
    output_dir: Path = config["run"]["output_dir"]
    problem = config["problem"]
    stages = [name for name in ("greedy", "point_swap", "annealing") if name in stage_results]
    rows = ["tau\tk\t" + "\t".join(stages) + "\tfinal"]
    blocks = []
    for tau in problem["tau"]:
        for k in problem["k"]:
            scores = {}
            for stage in stages:
                paths = [p for p in stage_results[stage].get((tau, k), []) if p.is_file()]
                if paths:
                    best = best_first(paths)[0]
                    scores[stage] = (hard_score(best), best)
            if not scores:
                raise PipelineError(f"No solution found for tau={tau:g}, k={k}")
            # Later steps never get worse than their input; prefer them on ties.
            best_stage = max(scores, key=lambda stage: (scores[stage][0], stages.index(stage)))
            text = scores[best_stage][1].read_text(encoding="utf-8")
            blocks.append(text if text.endswith("\n") else text + "\n")
            rows.append(
                f"{tau:g}\t{k}\t"
                + "\t".join(str(scores[stage][0]) if stage in scores else "-" for stage in stages)
                + f"\t{scores[best_stage][0]}"
            )
    (output_dir / "submission.txt").write_text("".join(blocks), encoding="utf-8")
    (output_dir / "scores.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\nHard score (number of serviced buildings) after each step:\n")
    for row in rows:
        print("  " + "  ".join(f"{cell:>10}" for cell in row.split("\t")))
    print(f"\nSubmission: {output_dir / 'submission.txt'}")
    print(f"Scores:     {output_dir / 'scores.tsv'}")


def existing_results(config: dict, step: str) -> Results:
    """Solutions left by an earlier run of step 2 or step 3."""

    base = config["run"]["output_dir"] / step
    results: Results = {}
    for tau in config["problem"]["tau"]:
        for k in config["problem"]["k"]:
            name = subproblem_name(tau, k)
            if step == "step2_greedy":
                paths = list((base / "candidates" / name).glob("*.txt"))
            else:
                paths = list((base / name).glob("*/best_solution.txt"))
            if paths:
                results[(tau, k)] = best_first(paths)
    return results


# --------------------------------------------------------------------------- #
def parse_steps(text: str) -> list[int]:
    try:
        steps = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError:
        raise PipelineError(f"--steps must look like 1,2,3,4; got {text!r}") from None
    if not steps or any(step not in STEP_NAMES for step in steps):
        raise PipelineError("--steps may only contain 1, 2, 3 and 4")
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    parser.add_argument("--steps", default="1,2,3,4", help="Comma-separated steps to run (default: 1,2,3,4)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config.resolve())
        steps = parse_steps(args.steps)
        if not config["step3_point_swap"]["enabled"]:
            steps = [step for step in steps if step != 3]
        if not config["step4_annealing"]["enabled"]:
            steps = [step for step in steps if step != 4]
        runner = Runner(config["run"])
        output_dir: Path = config["run"]["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config_used.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
        )

        problem = config["problem"]
        print(f"GeoJSON:   {problem['geojson']}")
        print(f"tau:       {', '.join(f'{tau:g}' for tau in problem['tau'])}")
        print(f"k:         {', '.join(map(str, problem['k']))}")
        print(f"Workers:   {runner.workers}")
        print(f"Output:    {output_dir}")
        print(f"Steps:     {', '.join(map(str, steps))}")
        greedy = config["step2_greedy"]
        if greedy["method"] == "two_step":
            combos = len(greedy["alpha"]) * len(greedy["lookahead_width"])
            print(f"Greedy:    two_step, {combos} parameter set(s) per sub-problem")
        else:
            print(f"Greedy:    multistart, {len(greedy['alpha'])} alpha value(s)")
        subproblems = len(problem["tau"]) * len(problem["k"])
        budget = 0.0
        if 3 in steps and config["step3_point_swap"]["max_seconds"]:
            budget += (float(config["step3_point_swap"]["max_seconds"]) * subproblems
                       * int(config["step3_point_swap"]["solutions"]))
        if 4 in steps:
            budget += float(config["step4_annealing"]["max_seconds"]) * subproblems
        if budget:
            print(f"Time limits of steps 3-4 add up to at most {format_seconds(budget)} "
                  f"(plus steps 1-2 and loading time).")

        stage_results: dict[str, Results] = {}
        timings: list[tuple[str, float]] = []

        def begin(step: int) -> float:
            print(f"\n=== Step {step}: {STEP_NAMES[step]} ===", flush=True)
            return time.monotonic()

        if 1 in steps:
            started = begin(1)
            segments, candidates = step1_visibility(config, runner)
            timings.append((STEP_NAMES[1], time.monotonic() - started))
        else:
            segments, candidates = require_visibility(config)

        if 2 in steps:
            started = begin(2)
            stage_results["greedy"] = step2_greedy(config, runner, segments, candidates)
            timings.append((STEP_NAMES[2], time.monotonic() - started))
        else:
            stage_results["greedy"] = existing_results(config, "step2_greedy")
            if not stage_results["greedy"]:
                raise PipelineError("Step 2 output is missing; include step 2 in --steps")

        if 3 in steps:
            started = begin(3)
            stage_results["point_swap"] = step3_point_swap(config, runner, segments, stage_results["greedy"])
            timings.append((STEP_NAMES[3], time.monotonic() - started))
        elif config["step3_point_swap"]["enabled"]:
            found = existing_results(config, "step3_point_swap")
            if found:
                stage_results["point_swap"] = found

        if 4 in steps:
            started = begin(4)
            inputs = stage_results.get("point_swap") or stage_results["greedy"]
            stage_results["annealing"] = step4_annealing(config, runner, segments, inputs)
            timings.append((STEP_NAMES[4], time.monotonic() - started))

        write_summary(config, stage_results)
        if timings:
            print("\nTime per step:")
            for name, seconds in timings:
                print(f"  {name:<18} {format_seconds(seconds)}")
    except PipelineError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
