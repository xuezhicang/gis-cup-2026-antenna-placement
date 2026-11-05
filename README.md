# GIS Cup 2026: antenna placement

Code for the [ACM SIGSPATIAL GIS Cup 2026](https://sigspatial2026.sigspatial.org/giscup.html).

**Problem.** Given building footprints, place `k` antennas on building
boundaries. A building is *serviced* when at least a fraction `tau` of its
perimeter is visible from the antennas (a sight line may touch a boundary but
not pass through a building). Maximise the number of serviced buildings for
every combination of `tau` and `k`.

## Method

The pipeline has four steps. Steps 2–4 run once per `(tau, k)` sub-problem.

```
config.toml
    │
    ▼
Step 1  Candidate points and visibility (run once, cached)
    │   Candidates: every building corner, plus a point every edge_spacing metres
    │   along each edge. For each candidate, compute which parts of which
    │   building edges it can see.
    ▼
Step 2  Greedy construction                for each (tau, k)
    │   Add antennas one at a time, choosing with a two-step lookahead
    │   (or: many greedy runs from different first antennas).
    │   Optionally several parameter sets -> several solutions.
    ▼
Step 3  One-point swap
    │   Repeatedly replace one antenna by the best unselected candidate
    │   while the score strictly improves (for the best N solutions).
    ▼
Step 4  Simulated annealing
    │   Pick seeds by score and diversity, then run random 1-swap, 2-swap and
    │   ruin-and-recreate moves that may temporarily lose score; the best
    │   solution found is always kept.
    ▼
submission.txt + scores.tsv
```

The score that counts (the *hard score*) is flat most of the time: lifting a
building from 10% to 30% coverage earns nothing until it reaches `tau`. To give
the search a signal, steps 2–4 also use a *soft score*,
`sum over buildings of min(coverage / tau, 1)`, which rewards partial
progress. Coverage is kept up to date incrementally by a C++ extension, so
evaluating a candidate antenna is fast.

### Step 2: two greedy methods

Both methods rank a candidate by `alpha × hard score + (1 − alpha) × soft
score` (each normalised). Choose one with `[step2_greedy] method`; the choice
applies to every `k` in the config.

- **`two_step`** (default). One greedy run from an empty solution. At each
  step it takes the `lookahead_width` best candidates, tries the
  `lookahead_width` best second antennas after each of them, and adds the
  first antenna of the best pair. One step costs about `1 + lookahead_width`
  passes over all candidates, so this works for any `k`.
- **`multistart`**. Every candidate is first scored on its own. A set of
  starting antennas is then chosen: the `grid_top_k` best candidates in every
  `grid_size_meters` grid cell, optionally plus the best `global_top_fraction`
  of all candidates and `random_count` random ones. From each start, plain
  greedy (no lookahead) adds the other `k − 1` antennas, and the best result
  is kept. When `k` is small the first antenna matters a lot, so trying many
  starts helps. However, the cost is roughly *number of starts × one greedy
  run*, so it is only practical for small `k` (about 20 or less). The
  competition run used it for `k = 9`.

### Several solutions and seed selection

With `run.py`'s built-in defaults (and in `config_quick.toml`), every
sub-problem follows a single path: one greedy solution, one swap run, one
annealing start. `config.toml` widens it in three places (6 parameter sets,
4 swapped solutions, 2 seeds):

1. **Step 2, several parameter sets.** Give `alpha` and/or `lookahead_width`
   as lists, e.g. `alpha = [0.4, 0.6, 0.8, 1.0]` and
   `lookahead_width = [1, 3, 5]`. Every combination gives one solution (12
   here). On one core they run one after another; with `workers > 1` they are
   shared between the processes. Step 2 takes correspondingly longer, and
   for large `k` this adds up quickly.
2. **Step 3, `solutions = N`.** The best `N` distinct step-2 solutions are
   each improved by one-point swap, one after another.
3. **Step 4, `seeds = S`.** `S` starting solutions are picked from the step-3
   results with the rule used for the competition submission: the best
   solution first, then repeatedly the solution with the highest
   `0.6 × score quality + 0.4 × building diversity`. *Building diversity* is
   one minus the largest Jaccard similarity between its serviced buildings
   and those of a seed already chosen. Only solutions within
   `seed_score_tolerance` (5%) of the best score are considered, and the
   limit is relaxed if too few qualify; identical antenna sets count once.
   The processes are shared equally, so each seed gets `workers / S`
   temperature profiles, and `S` must divide `workers`. The chosen seeds are
   listed in `step4_annealing/<sub-problem>_seeds.tsv`.

A small test on 100 buildings (6 sub-problems, `k = 3` and `9`, 2 processes,
20 s of annealing each) gave these total hard scores over all sub-problems:

| Setting | After step 2 | After step 3 | Final |
|---|---|---|---|
| 1 parameter set, 1 seed | 274 | 276 | 290 |
| 12 parameter sets, 1 seed | 279 | 286 | 290 |
| 12 parameter sets, 2 seeds | 279 | 286 | 286 |

More parameter sets clearly improved steps 2 and 3, but annealing closed most
of the gap within 20 s. With only 2 processes, 2 seeds left each seed a single
temperature (×1.0) instead of ×0.5 and ×2.0, and that run ended slightly
lower. These are single short runs, so differences of a few buildings are
within random variation. Several seeds make most sense when `workers` is large
enough to give every seed several temperatures.

### Step 4: temperatures in simulated annealing

Every process runs its own annealing search for the full `max_seconds`,
starting from its seed (the step-3 solution when `seeds = 1`); the best result
of all processes is kept.

- **Temperature profiles: workers ÷ seeds per seed.** Every process runs one
  seed at one temperature, so `workers = seeds × temperatures`. You do not set
  the number of temperatures yourself; `run.py` computes it. Each profile
  multiplies the starting temperature by a factor between ×0.5 (cooler, stays
  close to good solutions) and ×2.0 (hotter, explores more), evenly spaced on
  a log scale (`temperature_log_spread = 0.7` sets the range). Every seed gets
  the same set:

  | `workers` | `seeds` | Temperatures per seed | Multipliers |
  |---|---|---|---|
  | 1 | 1 | 1 | ×1.00 |
  | 4 | 1 | 4 | ×0.50, 0.79, 1.26, 2.01 |
  | 4 | 2 | 2 | ×0.50, 2.01 (`config.toml`) |
  | 4 | 4 | 1 | ×1.00 |
  | 8 | 2 | 4 | ×0.50, 0.79, 1.26, 2.01 |

  With a fixed number of workers, you trade seeds against temperatures: more
  seeds start from more different solutions, more temperatures search each
  start in more ways.
- **Fewer distinct solutions than seeds.** If a sub-problem has fewer
  distinct solutions than `seeds` (for example when several parameter sets
  lead to the same antennas), `run.py` uses fewer seeds and gives each of
  them more temperatures, and prints a note. With `workers = 4`, `seeds = 2`
  and only one distinct solution, that solution gets all 4 temperatures.
- **Calibrated starting temperature.** At the start of a cycle, losing one
  serviced building is accepted with probability 10%, falling to 0.01% at the
  end. Soft-score losses are scaled from a few trial moves: a typical loss is
  accepted with probability 50% at the start and 1% at the end.
- **Cooling counts iterations, not seconds.** The temperature falls
  geometrically over `cycle_iterations` iterations (default 20,000) and is
  then raised again for a new cycle. It is also raised early after
  `stagnation_iterations` iterations (default 20,000) without a new best
  solution.
- **Short runs.** If `max_seconds` is short, even one cooling cycle may not
  finish, and the search never reaches the cold end. The log line
  `FINISHED iteration=...` shows how many iterations ran. In our tests the
  speed was about 90–380 iterations per second, depending on the data and
  `k`. For short runs, lower `cycle_iterations` and `stagnation_iterations`
  so that a few cycles fit. In one test (1000 buildings, `tau = 0.49`,
  `k = 49`, 60 s, about 6,000 iterations, 3 random seeds), the default
  reached 549, 549 and 554 serviced buildings, while 3000 reached 555, 554
  and 554. `config_quick.toml` uses:

  ```toml
  [step4_annealing.advanced]
  cycle_iterations = 3000
  stagnation_iterations = 3000
  ```

## Installation

Works on Linux and macOS. First get the code:

```bash
git clone https://github.com/xuezhicang/gis-cup-2026-antenna-placement.git
cd gis-cup-2026-antenna-placement
```

Then create a Python environment in one of two ways.

### Option A: conda / mamba (recommended)

`environment.yml` installs everything the project needs, including a C++
compiler and MPI for parallel runs (`workers > 1`). If you have neither conda
nor mamba, install [Miniforge](https://github.com/conda-forge/miniforge),
which includes both.

```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
mamba activate gis-cup-2026            # or: conda activate gis-cup-2026
```

### Option B: pip and venv

Fewer packages to install, and enough for `workers = 1`. Needs Python 3.10 or
newer and a C++17 compiler (macOS: `xcode-select --install`; Ubuntu/Debian:
`sudo apt install build-essential python3-venv`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For `workers > 1` with this option, also install MPI and `mpi4py`:

```bash
# macOS: brew install open-mpi     Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev
python3 -m pip install mpi4py
```

### Then, for both options

With the environment activated:

```bash
bash build_cpp.sh            # compiles the C++ extensions into build/
bash data/download_data.sh   # downloads the sample and competition datasets
```

Activate the environment again in every new terminal
(`mamba activate gis-cup-2026`, or `source .venv/bin/activate` for option B)
before running anything. If you switch environments or Python versions, run
`bash build_cpp.sh` again, because the compiled extensions only work with the
Python they were built for.

## Quick start

There are two ready-made settings files:

| File | Purpose | Needs MPI | Time |
|---|---|---|---|
| `config_quick.toml` | Check that everything works: sample data, first 1000 candidates, 2 sub-problems, one core | No | About 2 minutes |
| `config.toml` | The full method on the competition problem: 6 greedy parameter sets, 4 swapped solutions and 2 seeds per sub-problem, 4 processes | Yes | About 6 hours (estimate, see below) |

First run the quick check:

```bash
python3 run.py --config config_quick.toml
```

Then run the full problem:

```bash
python3 run.py
```

Without MPI, or to use a single core, set `workers = 1` and `seeds = 1` in
`config.toml` (seeds must divide workers). For a much faster single run, also
set `alpha = 0.8`, `lookahead_width = 3` and `solutions = 1`.

## Configuration

All settings are in `config.toml`; every key is commented there. The ones
you are most likely to change:

| Section | Key | Meaning |
|---|---|---|
| `[problem]` | `geojson`, `tau`, `k` | The input data and the sub-problems (every `tau` × `k` pair) |
| `[run]` | `workers` | Number of parallel processes (1 = no MPI needed) |
| `[step1_visibility]` | `edge_spacing` | Distance between candidates along an edge, in metres. Smaller is better but slower |
| `[step2_greedy]` | `method`, `alpha`, `lookahead_width` | Greedy variant and its settings; `alpha` and `lookahead_width` may be lists |
| `[step3_point_swap]` | `solutions`, `max_seconds`, `max_generations` | How many greedy solutions to improve, and time and iteration limits for each |
| `[step4_annealing]` | `max_seconds`, `seeds` | Simulated-annealing time per sub-problem, and the number of seeds |

Steps 3 and 4 can be switched off with `enabled = false`. Unknown keys are
reported as errors, so a typo cannot be silently ignored.

`run.py --steps` reruns only some steps, reusing earlier results. For example,
after changing only the annealing time:

```bash
python3 run.py --steps 4
```

## Parallel runs (`workers`)

`workers` is used by every step, but each step splits the work differently:

| Step | How processes share the work | Effect of more workers |
|---|---|---|
| 1. Visibility | Candidates are split evenly | Almost proportional speedup |
| 2. Greedy, `two_step` | Each process takes whole runs (one sub-problem with one parameter set) | Runs go side by side; at most one process per run, and the large-`k` runs set the total time |
| 2. Greedy, `multistart` | Process 0 hands out starts, the others compute | Needs `workers ≥ 3` to be faster than `workers = 1` |
| 3. One-point swap | Solutions are improved in turn; within one, the trial removals are split | Faster per solution, up to `k` processes |
| 4. Annealing | Sub-problems run in turn; the processes are split over the seeds, each a separate search at its own temperature | Same run time, but more searching and usually better results |

In steps 2–4 every process loads its own copy of the visibility table, so
memory use is roughly `workers` × the table size. Memory usually limits how
many workers you can use.

## Output

```
outputs/
├── config_used.json          the complete settings of the last run
├── cache/visibility_.../     step 1 result, reused while the GeoJSON and step-1 settings are unchanged
├── step2_greedy/candidates/  every greedy solution, per sub-problem
├── step2_greedy/best/        the best greedy solution per sub-problem
├── step3_point_swap/         one folder per improved solution, plus best/
├── step4_annealing/          annealing runs, and <sub-problem>_seeds.tsv with the chosen seeds
├── submission.txt            the best solution of every sub-problem, in competition format
└── scores.tsv                hard score after each step
```

Each solution is three lines: `(tau, k)`, the antenna coordinates, and the IDs
of the serviced buildings. `submission.txt` can be checked with the official
evaluator, <https://github.com/alowe/gis-cup-2026-evaluator>.

## Running time and memory

Step 1 and the greedy step take most of the time, and both grow with the
number of candidate points. Measured on the **sample dataset** at
`edge_spacing = 20` (86,542 candidates) on a 2-core cloud machine:

| Step | Measured |
|---|---|
| 1. Visibility | about 34 ms per candidate and process: 26 min with `workers = 2` |
| 2. Two-step greedy (`lookahead_width = 5`) | about 1.5 s per antenna: k = 50 took 1.5 min |
| 3. One-point swap | about 5 s per generation for k = 50 with 2 processes |
| 4. Simulated annealing | exactly `max_seconds` per sub-problem |
| Visibility table on disk | 1.25 GB |
| Memory | about 0.8 GB per process |

Expect these numbers to scale roughly with the number of candidates. The
**competition dataset** has 334,000 candidates at 20 m (3.9 × the sample) and
813,000 at 5 m (9.4 ×). At 20 m that means roughly 5 GB of disk, 3 GB of
memory per process, about 3 CPU-hours for step 1 (divided by `workers`), and
around 45 minutes of greedy per `tau` value for k = 484.

For `config.toml` as shipped (4 workers, 6 parameter sets, 4 swapped
solutions, 2 seeds, 600 s of annealing), the same scaling gives roughly
45 minutes for step 1, 2 hours for step 2, up to 1.2 hours for step 3 and
1.5 hours for step 4: about **6 hours** in total, with about **12 GB** of
memory. These are estimates from the sample measurements, not measured on the
competition data.

To go faster: use more `workers` (if memory allows), a larger
`edge_spacing`, fewer or smaller `lookahead_width` values, fewer `alpha`
values, or fewer `solutions`. `config_quick.toml` finishes in a few minutes.

Step 1 is cached, so it only runs again when the GeoJSON or the step-1
settings change.

## Repository layout

```
├── run.py                      runs all steps from config.toml
├── config.toml                 main settings
├── config_quick.toml           small test run
├── environment.yml             conda / mamba environment
├── requirements.txt            pip requirements
├── build_cpp.sh                compiles the C++ extensions
├── cpp/                        C++ extensions (visibility engine, coverage scorer, building KD-tree)
├── common/                     code shared by the steps
├── step1_visibility/           step 1
├── step2_greedy/               step 2 (two-step lookahead and multistart greedy)
├── step3_point_swap/           step 3
├── step4_annealing/            step 4 (seed selection and simulated annealing)
├── tools/                      export helpers for inspecting results in GIS software
├── tests/                      unit tests
├── data/                       download script and data licence
└── LICENSE                     MIT licence for the code
```

Every step script can also be run on its own; see `--help`, for example
`python3 step3_point_swap/beam_point_swap.py --help`. `run.py` prints the exact
command it uses for each step.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

## License

The code is released under the [MIT License](LICENSE).

The datasets are not part of this repository. They are derived from
Microsoft's Global ML Building Footprints and licensed under
CDLA-Permissive-2.0; see [data/README.md](data/README.md).
