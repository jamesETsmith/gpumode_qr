"""Plot QR benchmark results over time and visualize variant/branch history.

This module is intentionally torch-free: it only parses the JSON results in
``db/`` and the local git history, then renders figures with matplotlib. It is
meant to run on the host (not inside the ROCm container).

Run it via the thin CLI wrapper::

    uv run --with matplotlib python scripts/plot_results.py

See :func:`generate_all` for the top-level entry point.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

BASELINE_IMPL = "torch_geqrf"


@dataclass
class ShapeResult:
    name: str
    n: int
    batch: int
    cond: int
    median_ms: float
    min_ms: float
    passed: bool


@dataclass
class Run:
    """One benchmark run (one db/*.json file)."""

    impl: str
    git_commit: str  # normalized (``-dirty`` suffix stripped)
    raw_commit: str  # as written in the file
    dirty: bool
    date: datetime
    path: Path
    shapes: dict[str, ShapeResult] = field(default_factory=dict)
    all_pass: bool | None = None


def _strip_dirty(commit: str) -> tuple[str, bool]:
    commit = (commit or "").strip()
    if commit.endswith("-dirty"):
        return commit[: -len("-dirty")], True
    return commit, False


def _parse_date(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z``."""
    if not value:
        return datetime.min
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min
    # Drop tz info so all timestamps are comparable / naive for plotting.
    return dt.replace(tzinfo=None)


def load_results(db_dir: str | Path) -> list[Run]:
    """Load and parse every ``db/*.json`` file, sorted by date (ascending).

    Robust to malformed files and the ``-dirty`` commit suffix; files that
    cannot be parsed are skipped with a warning rather than aborting.
    """
    db_dir = Path(db_dir)
    runs: list[Run] = []
    for path in sorted(db_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"WARNING: skipping unreadable {path.name}: {exc}")
            continue

        commit, dirty = _strip_dirty(data.get("git_commit", ""))
        run = Run(
            impl=data.get("impl", path.stem),
            git_commit=commit,
            raw_commit=data.get("git_commit", ""),
            dirty=dirty,
            date=_parse_date(data.get("date", "")),
            path=path.resolve(),
            all_pass=data.get("extra", {}).get("all_benchmarks_pass"),
        )
        for entry in data.get("benchmark_results", []):
            shape = entry.get("shape", {})
            timing = entry.get("timing", {})
            correctness = entry.get("correctness", {})
            name = entry.get("name") or shape.get("name", "?")
            median = timing.get("median_ms")
            if median is None:
                continue
            run.shapes[name] = ShapeResult(
                name=name,
                n=int(shape.get("n", 0)),
                batch=int(shape.get("batch", 0)),
                cond=int(shape.get("cond", 0)),
                median_ms=float(median),
                min_ms=float(timing.get("min_ms", median)),
                passed=bool(correctness.get("passed", True)),
            )
        runs.append(run)

    runs.sort(key=lambda r: r.date)
    return runs


# ---------------------------------------------------------------------------
# Structured variant history helper
# ---------------------------------------------------------------------------


@dataclass
class VariantHistory:
    impl: str
    first_seen: datetime
    commits: list[str]  # normalized commit hashes, ordered
    best_median_ms: dict[str, float]  # shape name -> best (min) median
    runs: int


def build_variant_history(runs: Iterable[Run]) -> dict[str, VariantHistory]:
    """Summarize results per variant.

    Returns, for each ``impl``: the first date it appears, the ordered list of
    commit hashes that produced results for it, and the best (lowest) median per
    benchmark shape. Reusable by plots and future tooling.
    """
    hist: dict[str, VariantHistory] = {}
    for run in sorted(runs, key=lambda r: r.date):
        vh = hist.get(run.impl)
        if vh is None:
            vh = VariantHistory(
                impl=run.impl,
                first_seen=run.date,
                commits=[],
                best_median_ms={},
                runs=0,
            )
            hist[run.impl] = vh
        vh.runs += 1
        if run.git_commit and run.git_commit not in vh.commits:
            vh.commits.append(run.git_commit)
        for name, sr in run.shapes.items():
            prev = vh.best_median_ms.get(name)
            if prev is None or sr.median_ms < prev:
                vh.best_median_ms[name] = sr.median_ms
    return hist


def _baseline_medians(runs: Iterable[Run]) -> dict[str, float]:
    """Best (lowest) median per shape for the baseline impl, for speedups."""
    medians: dict[str, float] = {}
    for run in runs:
        if run.impl != BASELINE_IMPL:
            continue
        for name, sr in run.shapes.items():
            prev = medians.get(name)
            if prev is None or sr.median_ms < prev:
                medians[name] = sr.median_ms
    return medians


# ---------------------------------------------------------------------------
# Leaderboard: cross-shape geomean ranking metric
# ---------------------------------------------------------------------------
#
# The GPUMODE challenge ranks passing submissions "by runtime using the
# geometric mean of benchmark cases" (AGENTS.md). We reproduce that ranking
# number here from the local db so it can be compared against the GPUMODE
# leaderboard without re-running any GPU benchmarks.
#
# Assumption: we use each shape's per-shape ``median_ms`` (10 timed runs) as
# that benchmark case's runtime, and take the geometric mean across the shapes.


def _geomean(values: Iterable[float]) -> float:
    """Geometric mean of positive values (``nan`` if empty)."""
    vals = list(values)
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _expected_shape_names(runs: Iterable[Run]) -> set[str]:
    """Canonical benchmark-shape set = union of shape names across all runs.

    All runs benchmark the same ``inputs.BENCHMARK_SHAPES`` (7 shapes), so the
    union is the full set a "complete" run is expected to cover.
    """
    names: set[str] = set()
    for run in runs:
        names.update(run.shapes.keys())
    return names


def _latest_run_per_variant(runs: Iterable[Run]) -> dict[str, Run]:
    """Most recent (by date) db run for each impl."""
    latest: dict[str, Run] = {}
    for run in runs:
        prev = latest.get(run.impl)
        if prev is None or run.date >= prev.date:
            latest[run.impl] = run
    return latest


@dataclass
class LeaderboardRow:
    impl: str
    geomean_median_ms: float
    n_shapes: int
    n_expected: int
    complete: bool
    speedup_vs_baseline: float | None  # baseline_geomean / this_geomean
    date: datetime


def build_leaderboard(runs: Iterable[Run]) -> list[LeaderboardRow]:
    """Per-variant leaderboard from the latest db run of each variant.

    For each impl, take its most recent db file, compute the geometric mean of
    the per-shape ``median_ms`` across the benchmark shapes, and the speedup vs
    the ``torch_geqrf`` baseline's geomean (over the same complete shape set).

    A row is only ranked (and given a speedup) if the run covers all expected
    shapes; partial runs are still returned but flagged ``complete=False`` so
    the caller can note them separately. Rows are sorted fastest-first by
    geomean, with complete runs ranked ahead of partial ones.
    """
    runs = list(runs)
    expected = _expected_shape_names(runs)
    n_expected = len(expected)
    latest = _latest_run_per_variant(runs)

    # Baseline geomean over the complete shape set (from the baseline's latest
    # complete run), used as the speedup reference.
    baseline_geomean: float | None = None
    base_run = latest.get(BASELINE_IMPL)
    if base_run is not None and expected.issubset(base_run.shapes.keys()):
        baseline_geomean = _geomean(base_run.shapes[name].median_ms for name in expected)

    rows: list[LeaderboardRow] = []
    for impl, run in latest.items():
        names = set(run.shapes.keys())
        complete = expected.issubset(names) and n_expected > 0
        if complete:
            gm = _geomean(run.shapes[name].median_ms for name in expected)
        else:
            # geomean over whatever shapes exist, for context only
            gm = _geomean(sr.median_ms for sr in run.shapes.values())
        speedup = None
        if complete and baseline_geomean is not None and gm > 0:
            speedup = baseline_geomean / gm
        rows.append(
            LeaderboardRow(
                impl=impl,
                geomean_median_ms=gm,
                n_shapes=len(names),
                n_expected=n_expected,
                complete=complete,
                speedup_vs_baseline=speedup,
                date=run.date,
            )
        )

    rows.sort(key=lambda r: (not r.complete, r.geomean_median_ms))
    return rows


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    """Render :func:`build_leaderboard` output as a compact text table."""
    lines: list[str] = []
    lines.append(
        "Leaderboard (geomean of per-shape median_ms; lower is better; speedup vs torch_geqrf):"
    )
    header = f"  {'variant':<30} {'geomean(median) ms':>20} {'speedup':>10}  shapes"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        if r.complete:
            speed = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "-"
            shapes = f"{r.n_shapes}/{r.n_expected}"
        else:
            speed = "(partial)"
            shapes = f"{r.n_shapes}/{r.n_expected}*"
        lines.append(f"  {r.impl:<30} {r.geomean_median_ms:>20.4f} {speed:>10}  {shapes}")
    lines.append("  * partial: run does not cover all benchmark shapes; not ranked.")
    lines.append("  assumption: per-shape median_ms is the case runtime for the geomean.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git DAG parsing
# ---------------------------------------------------------------------------


@dataclass
class Commit:
    sha: str
    short: str
    parents: list[str]
    refs: str
    date: datetime
    subject: str
    lane: str = "main"


@dataclass
class GitDag:
    commits: dict[str, Commit]  # sha -> Commit
    order: list[str]  # shas, newest-first (git log order)
    lanes: list[str]  # lane (branch) names, main first


def _git(args: list[str], repo: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL)


def load_git_dag(repo: str | Path) -> GitDag | None:
    """Reconstruct the commit DAG and assign each commit to a branch lane.

    Lane assignment: main's first-parent history is the trunk (lane ``main``).
    Every other local branch claims the commits unique to its line of
    development (its ``rev-list`` minus the trunk). This surfaces the
    ``variant/<name>`` development lanes and their merge points into main.
    """
    repo = Path(repo)
    try:
        sep = "\x1f"
        raw = _git(
            [
                "log",
                "--all",
                "--date-order",
                f"--pretty=format:%H{sep}%h{sep}%P{sep}%D{sep}%cI{sep}%s",
            ],
            repo,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    commits: dict[str, Commit] = {}
    order: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split(sep)
        if len(parts) < 6:
            continue
        sha, short, parents, refs, cdate, subject = parts[:6]
        commits[sha] = Commit(
            sha=sha,
            short=short,
            parents=parents.split() if parents.strip() else [],
            refs=refs,
            date=_parse_date(cdate),
            subject=subject,
        )
        order.append(sha)

    if not commits:
        return None

    # Enumerate local branches (main first, then the rest).
    try:
        branch_out = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads"], repo)
    except subprocess.CalledProcessError:
        branch_out = ""
    branches = [b.strip() for b in branch_out.splitlines() if b.strip()]

    def _revlist(ref: str) -> list[str]:
        try:
            return _git(["rev-list", ref], repo).split()
        except subprocess.CalledProcessError:
            return []

    # main's first-parent history is the trunk; those commits are lane "main"
    # regardless of which other branches also contain them.
    trunk: set[str] = set()
    if "main" in branches:
        trunk = set(_git(["rev-list", "--first-parent", "main"], repo).split())
        for sha in trunk:
            if sha in commits:
                commits[sha].lane = "main"

    # For every other commit, the owning branch is the *most specific* branch
    # that contains it -- i.e. the one with the smallest rev-list. This keeps
    # merged variant commits on their variant lane instead of leaking onto
    # unrelated descendant branches (e.g. this tooling branch) that also
    # contain them by virtue of descending from main.
    branch_sets: dict[str, set[str]] = {b: set(_revlist(b)) for b in branches}
    branch_size = {b: len(s) for b, s in branch_sets.items()}

    for sha, commit in commits.items():
        if sha in trunk:
            continue
        candidates = [b for b in branches if b != "main" and sha in branch_sets[b]]
        if candidates:
            commit.lane = min(candidates, key=lambda b: (branch_size[b], b))
        elif "main" in branch_sets and sha in branch_sets["main"]:
            commit.lane = "main"
        else:
            commit.lane = "other"  # detached / dangling

    # Order lanes: main first, then by the earliest commit date on the lane.
    used = {c.lane for c in commits.values()}
    earliest: dict[str, datetime] = {}
    for c in commits.values():
        cur = earliest.get(c.lane)
        if cur is None or c.date < cur:
            earliest[c.lane] = c.date
    lanes = ["main"] if "main" in used else []
    lanes += sorted((l for l in used if l != "main"), key=lambda l: earliest[l])

    return GitDag(commits=commits, order=order, lanes=lanes)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _shape_order(runs: Iterable[Run]) -> list[str]:
    """Benchmark shape names ordered by matrix size (n), then batch."""
    seen: dict[str, tuple[int, int]] = {}
    for run in runs:
        for name, sr in run.shapes.items():
            seen.setdefault(name, (sr.n, sr.batch))
    return sorted(seen, key=lambda k: seen[k])


def _variant_style(impls: list[str]):
    """Stable color+marker per impl. Baseline gets a fixed grey/dashed style."""
    import matplotlib as mpl

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    others = [i for i in impls if i != BASELINE_IMPL]
    cmap = mpl.colormaps["tab10"]
    colors = [cmap(i % cmap.N) for i in range(max(len(others), 1))]
    style: dict[str, dict] = {BASELINE_IMPL: dict(color="0.35", marker="o", linestyle="--")}
    for idx, impl in enumerate(others):
        style[impl] = dict(
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linestyle="-",
        )
    return style


def _shape_subtitle(runs: list[Run], shape_name: str) -> str:
    """``n=.., batch=.., cond=..`` string for a shape (empty if unknown)."""
    sr_example = next((r.shapes[shape_name] for r in runs if shape_name in r.shapes), None)
    if sr_example is None:
        return ""
    return f"n={sr_example.n}, batch={sr_example.batch}, cond={sr_example.cond}"


def _draw_shape_axis(
    ax,
    runs: list[Run],
    shape_name: str,
    impls: list[str],
    style: dict,
    x_of_run: dict,
    labels: list[str],
    *,
    label_fontsize: int = 6,
    annotate_fontsize: int = 6,
    marker_size: int = 7,
) -> None:
    """Draw a single shape's median_ms-over-runs series onto ``ax``.

    Shared by the combined small-multiples figure and the per-shape figures so
    both stay visually consistent (baseline dashed reference, per-point variant
    annotations, and red-x correctness-failure markers).
    """
    # baseline reference line for this shape
    base_vals = [
        r.shapes[shape_name].median_ms
        for r in runs
        if r.impl == BASELINE_IMPL and shape_name in r.shapes
    ]
    if base_vals:
        base = min(base_vals)
        ax.axhline(
            base,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=f"{BASELINE_IMPL} ref ({base:.2g} ms)",
        )

    for impl in impls:
        xs, ys, fail_x, fail_y = [], [], [], []
        for r in runs:
            if r.impl != impl or shape_name not in r.shapes:
                continue
            sr = r.shapes[shape_name]
            xs.append(x_of_run[id(r)])
            ys.append(sr.median_ms)
            if not sr.passed:
                fail_x.append(x_of_run[id(r)])
                fail_y.append(sr.median_ms)
        if not xs:
            continue
        st = style[impl]
        ax.plot(
            xs,
            ys,
            marker=st["marker"],
            color=st["color"],
            linestyle=st["linestyle"],
            markersize=marker_size,
            linewidth=1.4,
            label=impl,
            zorder=3,
        )
        # annotate each point with the variant name (compact)
        for x, y in zip(xs, ys):
            ax.annotate(
                impl.replace("blocked_", "b_"),
                (x, y),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=annotate_fontsize,
                color=st["color"],
                zorder=4,
            )
        # mark correctness failures distinctly
        if fail_x:
            ax.scatter(
                fail_x,
                fail_y,
                marker="x",
                color="red",
                s=110,
                linewidths=2.2,
                zorder=5,
                label="FAIL correctness",
            )

    ax.set_yscale("log")
    ax.set_ylabel("median (ms, log)", fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=label_fontsize, rotation=0)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.margins(x=0.08)
    # de-duplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    uniq = dict(zip(lbls, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=label_fontsize, loc="best")


def plot_performance_over_time(runs: list[Run], out_path: Path) -> Path:
    """Small-multiples: per shape, median_ms (log-y) vs run/commit order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = _shape_order(runs)
    impls = sorted({r.impl for r in runs})
    style = _variant_style(impls)

    # x position = chronological run index (commit order).
    x_of_run = {id(r): i for i, r in enumerate(runs)}
    labels = [f"{i}\n{r.date:%m-%d %H:%M}" for i, r in enumerate(runs)]

    ncols = 3
    nrows = (len(shapes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    for idx, shape_name in enumerate(shapes):
        ax = axes[idx // ncols][idx % ncols]
        _draw_shape_axis(ax, runs, shape_name, impls, style, x_of_run, labels)
        subtitle = _shape_subtitle(runs, shape_name)
        ax.set_title(f"{shape_name}\n{subtitle}", fontsize=9)

    # hide unused axes
    for idx in range(len(shapes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        "QR benchmark: median runtime per shape over runs (log-y, lower is better)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def plot_performance_per_shape(runs: list[Run], plots_dir: Path) -> list[Path]:
    """One standalone figure per benchmark shape.

    Writes ``plots/perf_<shape_name>.png`` for each shape: median_ms on a log-y
    axis versus chronological run/commit order, each variant a distinct
    color+marker with a legend, ``torch_geqrf`` as a dashed baseline reference,
    correctness failures marked with a red ``x``, and per-point variant
    annotations. Returns the list of absolute PNG paths written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = _shape_order(runs)
    impls = sorted({r.impl for r in runs})
    style = _variant_style(impls)

    x_of_run = {id(r): i for i, r in enumerate(runs)}
    labels = [f"{i}\n{r.date:%m-%d %H:%M}" for i, r in enumerate(runs)]

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for shape_name in shapes:
        fig, ax = plt.subplots(figsize=(max(9.0, 0.7 * len(runs) + 3.0), 5.5))
        _draw_shape_axis(
            ax,
            runs,
            shape_name,
            impls,
            style,
            x_of_run,
            labels,
            label_fontsize=8,
            annotate_fontsize=7,
            marker_size=8,
        )
        subtitle = _shape_subtitle(runs, shape_name)
        ax.set_title(
            f"{shape_name}  ({subtitle})\nmedian runtime over runs (log-y, lower is better)",
            fontsize=12,
        )
        ax.set_xlabel("run index (chronological / commit order)", fontsize=9)
        fig.tight_layout()
        out_path = plots_dir / f"perf_{shape_name}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        written.append(out_path.resolve())
    return written


def plot_branch_history(runs: list[Run], dag: GitDag, out_path: Path) -> Path:
    """Render branch lanes over time with merge points + result markers."""
    import matplotlib
    import matplotlib.dates as mdates

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lanes = dag.lanes
    lane_y = {lane: i for i, lane in enumerate(lanes)}

    impls = sorted({r.impl for r in runs})
    style = _variant_style(impls)
    baseline = _baseline_medians(runs)

    # results grouped by commit for annotation
    runs_by_commit: dict[str, list[Run]] = {}
    for r in runs:
        if r.git_commit:
            runs_by_commit.setdefault(r.git_commit, []).append(r)

    fig, ax = plt.subplots(figsize=(15, 2.0 + 1.1 * len(lanes)))

    # edges: connect each commit to its parents
    for commit in dag.commits.values():
        y0 = lane_y.get(commit.lane, 0)
        for parent in commit.parents:
            p = dag.commits.get(parent)
            if p is None:
                continue
            y1 = lane_y.get(p.lane, 0)
            ax.plot(
                [mdates.date2num(commit.date), mdates.date2num(p.date)],
                [y0, y1],
                color="0.75",
                linewidth=1.0,
                zorder=1,
            )

    # commit dots
    for commit in dag.commits.values():
        y = lane_y.get(commit.lane, 0)
        is_merge = len(commit.parents) >= 2
        ax.scatter(
            mdates.date2num(commit.date),
            y,
            s=140 if is_merge else 70,
            marker="D" if is_merge else "o",
            color="white",
            edgecolors="0.2",
            linewidths=1.6 if is_merge else 1.0,
            zorder=3,
        )
        ax.annotate(
            commit.short,
            (mdates.date2num(commit.date), y),
            textcoords="offset points",
            xytext=(0, 9 if not is_merge else 12),
            fontsize=6,
            color="0.35",
            ha="center",
            zorder=4,
        )
        if is_merge:
            ax.annotate(
                "merge",
                (mdates.date2num(commit.date), y),
                textcoords="offset points",
                xytext=(0, -13),
                fontsize=6,
                style="italic",
                color="0.4",
                ha="center",
                zorder=4,
            )

    # result markers on the commits that produced them
    labeled_impls: set[str] = set()
    for commit_sha, commit_runs in runs_by_commit.items():
        commit = dag.commits.get(commit_sha)
        if commit is None:
            continue  # result whose commit is not in the local DAG
        y = lane_y.get(commit.lane, 0)
        for offset, run in enumerate(commit_runs):
            st = style.get(run.impl, dict(color="black", marker="*"))
            # best per-shape speedup vs baseline (easy wins to surface)
            speedups = [
                baseline[name] / sr.median_ms
                for name, sr in run.shapes.items()
                if name in baseline and sr.median_ms > 0
            ]
            best = max(speedups) if speedups else None
            label = run.impl
            if best is not None and run.impl != BASELINE_IMPL:
                label = f"{run.impl} ({best:.2f}x)"
            ax.scatter(
                mdates.date2num(commit.date),
                y,
                s=180,
                marker=st["marker"],
                color=st["color"],
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
                label=run.impl if run.impl not in labeled_impls else None,
            )
            labeled_impls.add(run.impl)
            ax.annotate(
                label,
                (mdates.date2num(commit.date), y),
                textcoords="offset points",
                xytext=(6, -6 - 11 * offset),
                fontsize=7,
                fontweight="bold",
                color=st["color"],
                zorder=6,
            )

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=9)
    ax.set_ylim(-0.7, len(lanes) - 0.3)
    ax.invert_yaxis()  # main on top
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_xlabel("commit time (UTC-ish, local git timestamps)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title(
        "Variant/branch history: lanes = branches, diamonds = merges, "
        "colored markers = benchmark results (best per-shape speedup vs "
        f"{BASELINE_IMPL})",
        fontsize=11,
    )
    handles, lbls = ax.get_legend_handles_labels()
    uniq = dict(zip(lbls, handles))
    if uniq:
        ax.legend(
            uniq.values(),
            uniq.keys(),
            fontsize=8,
            loc="center left",
            title="benchmark result impl",
            framealpha=0.9,
        )
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def plot_geomean_over_iterations(runs: list[Run], out_path: Path) -> Path:
    """Leaderboard geomean improving over the research iterations.

    x-axis is chronological iteration order (each variant placed at the date of
    its latest db result, consistent with the per-shape perf figure's ordering);
    y-axis is the geometric mean of the 7 per-shape ``median_ms`` (log-y). Draws:

    - the ``torch_geqrf`` baseline geomean as a dashed reference line,
    - each variant's own geomean as a labeled scatter point,
    - the *best-so-far* geomean as a descending step line, and
    - an annotation on the final best with its speedup vs the baseline.

    Only variants with all benchmark shapes are shown (partial runs are excluded
    via :func:`build_leaderboard`'s ``complete`` flag).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = build_leaderboard(runs)
    baseline_row = next((r for r in rows if r.impl == BASELINE_IMPL), None)
    baseline_geomean = baseline_row.geomean_median_ms if baseline_row else None

    # Non-baseline complete variants, ordered chronologically by their latest run
    # date (research-iteration order); ties broken by geomean.
    variants = [r for r in rows if r.complete and r.impl != BASELINE_IMPL]
    variants.sort(key=lambda r: (r.date, r.geomean_median_ms))

    fig, ax = plt.subplots(figsize=(max(10.0, 0.85 * len(variants) + 3.0), 6.8))

    if not variants:
        ax.text(
            0.5,
            0.5,
            "no complete-shape variants to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path.resolve()

    xs = list(range(len(variants)))
    ys = [r.geomean_median_ms for r in variants]
    # Abbreviated variant names, reused for the x tick labels so each point is
    # self-identifying (avoids readers mistaking the tick index for an
    # "iteration number").
    tick_labels = [r.impl.replace("cholqr", "cqr").replace("blocked_", "b_") for r in variants]

    # best-so-far (cumulative minimum) as a step line
    best_so_far: list[float] = []
    cur = float("inf")
    for y in ys:
        cur = min(cur, y)
        best_so_far.append(cur)
    ax.step(
        xs,
        best_so_far,
        where="post",
        color="tab:blue",
        linewidth=1.8,
        zorder=2,
        label="best-so-far geomean",
    )

    # each variant's own geomean point (names now live on the x ticks, so no
    # per-point name annotations here to avoid double-clutter)
    ax.scatter(xs, ys, color="tab:orange", s=60, zorder=3, label="variant geomean")

    if baseline_geomean is not None:
        ax.axhline(
            baseline_geomean,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label=f"{BASELINE_IMPL} baseline ({baseline_geomean:.2f} ms)",
        )

    # annotate the final best with its speedup vs baseline
    best = variants[min(range(len(ys)), key=lambda i: ys[i])]
    if baseline_geomean is not None and best.geomean_median_ms > 0:
        speedup = baseline_geomean / best.geomean_median_ms
        bx = xs[variants.index(best)]
        ax.annotate(
            f"best: {best.impl}\n{best.geomean_median_ms:.2f} ms  ({speedup:.2f}x vs baseline)",
            (bx, best.geomean_median_ms),
            textcoords="offset points",
            xytext=(10, -34),
            fontsize=9,
            fontweight="bold",
            color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2),
            zorder=6,
        )

    ax.set_yscale("log")
    ax.set_ylabel("geomean of per-shape median_ms (log, lower is better)", fontsize=9)
    ax.set_xlabel("variant (chronological order; last = current champion)", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=40, ha="right")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.margins(x=0.06)
    ax.set_title(
        "Leaderboard geomean over research iterations "
        "(geomean of 7 per-shape medians; lower is better)",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def generate_all(repo: str | Path) -> list[Path]:
    """Load data + git history and write both figures under ``<repo>/plots``.

    Returns the list of absolute PNG paths written.
    """
    repo = Path(repo).resolve()
    db_dir = repo / "db"
    plots_dir = repo / "plots"

    runs = load_results(db_dir)
    if not runs:
        raise SystemExit(f"No results found under {db_dir}")

    written: list[Path] = []
    # Primary output: one standalone figure per benchmark shape.
    written.extend(plot_performance_per_shape(runs, plots_dir))
    # Kept as an extra overview: the combined small-multiples grid.
    written.append(plot_performance_over_time(runs, plots_dir / "perf_over_time.png"))
    # Leaderboard geomean improving over the research iterations.
    written.append(plot_geomean_over_iterations(runs, plots_dir / "geomean_over_iterations.png"))

    dag = load_git_dag(repo)
    if dag is not None:
        written.append(plot_branch_history(runs, dag, plots_dir / "branch_history.png"))
    else:
        print("WARNING: could not read git history; skipping branch figure")

    return written
