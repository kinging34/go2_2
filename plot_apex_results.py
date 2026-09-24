"""Plot and compare the TensorBoard logs written by the Go2 SNN APEX notebooks.

Usage:
    python plot_apex_results.py --runs-root runs --output apex_report
    python plot_apex_results.py --run runs/RUN_A --run runs/RUN_B --output apex_report

Install dependencies in the environment used to run this script:
    pip install tensorboard matplotlib numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TASKS = {"前進": "Forward", "後退": "Backward", "左移動": "Left", "左旋回": "Turn left"}
EVENT_GLOB = "events.out.tfevents.*"


@dataclass(frozen=True)
class Point:
    step: int
    value: float
    wall_time: float


@dataclass
class Run:
    path: Path
    label: str
    series: dict[str, list[Point]]

    def get(self, tag: str) -> list[Point]:
        return self.series.get(tag, [])


@dataclass
class Trace:
    path: Path
    label: str
    rows: list[dict[str, float | str]]


def normalize_label(path: Path) -> str:
    return re.sub(r"_\d{8}_\d{6}_.+$", "", path.name)


def discover_runs(root: Path, match: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {root}")
    paths = {p.parent.resolve() for p in root.rglob(EVENT_GLOB)}
    return sorted(p for p in paths if not match or match.lower() in p.name.lower())


def scalar_from_tensor(event, tensor_util) -> float:
    value = tensor_util.make_ndarray(event.tensor_proto)
    if value.size != 1:
        raise ValueError("Only scalar tensor summaries are supported")
    return float(value.reshape(-1)[0])


def load_run(path: Path, label: str) -> Run:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util import tensor_util
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required. Install it with: pip install tensorboard") from exc

    accumulator = EventAccumulator(
        str(path), size_guidance={"scalars": 0, "tensors": 0, "images": 0, "histograms": 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags()
    series: dict[str, list[Point]] = {}
    for tag in tags.get("scalars", []):
        series[tag] = [Point(int(e.step), float(e.value), float(e.wall_time))
                       for e in accumulator.Scalars(tag)]
    for tag in tags.get("tensors", []):
        if tag in series:
            continue
        values = []
        for e in accumulator.Tensors(tag):
            try:
                val = scalar_from_tensor(e, tensor_util)
            except (TypeError, ValueError):
                continue
            values.append(Point(int(e.step), val, float(e.wall_time)))
        if values:
            series[tag] = values

    # A resumed run may write the same step twice. Keep the newest value.
    clean = {}
    for tag, values in series.items():
        by_step = {}
        for point in values:
            if math.isfinite(point.value):
                if point.step not in by_step or point.wall_time >= by_step[point.step].wall_time:
                    by_step[point.step] = point
        if by_step:
            clean[tag] = [by_step[step] for step in sorted(by_step)]
    if not clean:
        raise ValueError(f"No scalar data found in {path}")
    return Run(path=path, label=label, series=clean)


def load_trace(path: Path) -> Trace:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation trace does not exist: {path}")
    required = {"step", "time_s", "task", "cmd_vx", "cmd_vy", "cmd_wz",
                "vx_body", "vy_body", "wz_body", "base_height",
                "reward_total", "reward_style", "reward_task", "active", "fallen"}
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trace {path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            parsed: dict[str, float | str] = {"task": row["task"]}
            for name in required - {"task"}:
                parsed[name] = float(row[name])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Trace CSV is empty: {path}")
    label = path.stem.split("_eval_trace_")[0]
    return Trace(path=path, label=label, rows=rows)


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def plot_lines(ax, runs: list[Run], tag: str, smooth: int, *, title: str,
               ylabel: str = "", color: str | None = None, raw: bool = False) -> bool:
    present = False
    for run in runs:
        values = run.get(tag)
        if not values:
            continue
        present = True
        x = [p.step / 1e6 for p in values]
        y = [p.value for p in values]
        if raw and smooth > 1:
            ax.plot(x, y, alpha=0.18, linewidth=0.9, color=color)
        ax.plot(x, moving_average(y, smooth), linewidth=1.8, label=run.label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Transitions (millions)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if present and len(runs) > 1:
        ax.legend(fontsize=8)
    if not present:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    return present


def plot_multi_tags(ax, runs: list[Run], tags: dict[str, str], smooth: int,
                    title: str, ylabel: str = "") -> bool:
    present = False
    for run in runs:
        for tag, short_name in tags.items():
            values = run.get(tag)
            if not values:
                continue
            present = True
            label = short_name if len(runs) == 1 else f"{run.label}: {short_name}"
            ax.plot([p.step / 1e6 for p in values],
                    moving_average([p.value for p in values], smooth),
                    linewidth=1.6, label=label)
    ax.set_title(title)
    ax.set_xlabel("Transitions (millions)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if present:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    return present


def save_figure(fig, outdir: Path, filename: str, plt) -> None:
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_charts(runs: list[Run], outdir: Path, smooth: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required. Install it with: pip install matplotlib") from exc

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    produced = []

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    found = [
        plot_lines(axs[0, 0], runs, "train/episode_total", smooth,
                   title="Training episode return", raw=True),
        plot_lines(axs[0, 1], runs, "train/episode_style", smooth,
                   title="Training style return", raw=True),
        plot_lines(axs[1, 0], runs, "train/episode_task", smooth,
                   title="Training task return", raw=True),
        plot_lines(axs[1, 1], runs, "train/episode_length", smooth,
                   title="Training episode length", ylabel="Steps", raw=True),
    ]
    if any(found):
        save_figure(fig, outdir, "01_training_returns.png", plt)
        produced.append("01_training_returns.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    found = []
    for ax, (tag, label) in zip(axs.flat, TASKS.items()):
        found.append(plot_lines(ax, runs, f"test/{tag}", smooth,
                                title=f"Prior OFF: {label} return", raw=True))
    if any(found):
        save_figure(fig, outdir, "02_prior_off_returns.png", plt)
        produced.append("02_prior_off_returns.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(18, 7), sharey="row")
    found = []
    for col, (tag, label) in enumerate(TASKS.items()):
        found.append(plot_lines(axs[0, col], runs, f"test/{tag}_survival", smooth,
                                title=f"{label}: survival", ylabel="Fraction" if col == 0 else ""))
        found.append(plot_lines(axs[1, col], runs, f"test/{tag}_tracking", smooth,
                                title=f"{label}: tracking", ylabel="Score" if col == 0 else ""))
        axs[0, col].set_ylim(-0.03, 1.03)
        axs[1, col].set_ylim(-0.03, 1.03)
    if any(found):
        save_figure(fig, outdir, "03_prior_off_quality.png", plt)
        produced.append("03_prior_off_quality.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    found = [
        plot_lines(axs[0], runs, "train/decap_factor_last", 1,
                   title="Action Prior during rollout", ylabel="Prior coefficient"),
        plot_lines(axs[1], runs, "train/prior_after_eval", 1,
                   title="Action Prior after evaluation", ylabel="Next coefficient"),
    ]
    if any(found):
        save_figure(fig, outdir, "04_action_prior.png", plt)
        produced.append("04_action_prior.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    found = [
        plot_multi_tags(axs[0, 0], runs, {"policy/kl": "accepted KL",
                                          "policy/max_kl": "maximum KL",
                                          "policy/rejected_kl": "rejected KL"},
                        smooth, "PPO divergence"),
        plot_lines(axs[0, 1], runs, "policy/lr", smooth,
                   title="PPO learning rate"),
        plot_multi_tags(axs[1, 0], runs, {"policy/epochs_used": "epochs",
                                          "policy/optimizer_steps": "optimizer steps"},
                        smooth, "PPO update coverage"),
        plot_lines(axs[1, 1], runs, "policy/early_stopped", smooth,
                   title="PPO early-stop frequency", ylabel="Fraction"),
    ]
    axs[0, 0].axhline(0.03, color="black", linestyle="--", linewidth=1,
                      alpha=0.5, label="hard stop = 0.03")
    axs[0, 1].set_yscale("log")
    axs[1, 1].set_ylim(-0.03, 1.03)
    if any(found):
        save_figure(fig, outdir, "05_ppo_health.png", plt)
        produced.append("05_ppo_health.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    found = [
        plot_multi_tags(axs[0, 0], runs, {"reward/tracking_lin": "linear",
                                          "reward/tracking_ang": "yaw"},
                        smooth, "Task tracking reward per step"),
        plot_multi_tags(axs[0, 1], runs, {"reward/imitation_angles": "joint angles",
                                          "reward/imitation_foot": "feet",
                                          "reward/imitation_quat": "orientation"},
                        smooth, "Style reward per step"),
        plot_multi_tags(axs[1, 0], runs, {"penalty/collision": "collision",
                                          "penalty/action_rate": "action rate",
                                          "penalty/feet_slip": "foot slip",
                                          "penalty/torque": "torque"},
                        smooth, "Penalties per step"),
        plot_multi_tags(axs[1, 1], runs, {"train/raw_collision_body_count": "raw collision bodies",
                                          "train/contact_only_rate": "contact-only fall rate",
                                          "train/fallen_rate": "actual fall rate"},
                        smooth, "Contact diagnostics"),
    ]
    if any(found):
        save_figure(fig, outdir, "06_reward_and_contacts.png", plt)
        produced.append("06_reward_and_contacts.png")
    else:
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    found = [
        plot_multi_tags(axs[0, 0], runs, {"snn/spike_rate_l1": "layer 1",
                                          "snn/spike_rate_l2": "layer 2",
                                          "snn/spike_rate_l3": "layer 3"},
                        smooth, "SNN spike rates"),
        plot_lines(axs[0, 1], runs, "policy/mean_abs", smooth,
                   title="Actor mean action magnitude"),
        plot_lines(axs[1, 0], runs, "policy/action_std", smooth,
                   title="Action standard deviation"),
        plot_lines(axs[1, 1], runs, "policy/decoder_gain_mean", smooth,
                   title="SNN decoder gain"),
    ]
    if any(found):
        save_figure(fig, outdir, "07_snn_actor.png", plt)
        produced.append("07_snn_actor.png")
    else:
        plt.close(fig)

    terrain_tags = sorted({tag for run in runs for tag in run.series if tag.startswith("terrain/")})
    if terrain_tags:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_multi_tags(ax, runs, {tag: tag.split("/", 1)[1] for tag in terrain_tags},
                        smooth, "Terrain curriculum level")
        save_figure(fig, outdir, "08_terrain_curriculum.png", plt)
        produced.append("08_terrain_curriculum.png")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    speed_found = False
    for run in runs:
        values = run.get("train/rollout_reward_per_step")
        if len(values) < 2:
            continue
        first = values[0]
        x = [(p.wall_time - first.wall_time) / 3600 for p in values]
        y = [(p.step - first.step) / 1e6 for p in values]
        axs[0].plot(x, y, label=run.label)
        speed_x = []
        speed_y = []
        for old, new in zip(values, values[1:]):
            dt = new.wall_time - old.wall_time
            ds = new.step - old.step
            if dt > 0 and ds > 0:
                speed_x.append(new.step / 1e6)
                speed_y.append(ds / dt)
        if speed_x:
            axs[1].plot(speed_x, moving_average(speed_y, smooth), label=run.label)
        speed_found = True
    for ax, title, xlabel, ylabel in ((axs[0], "Progress over wall time", "Hours", "Million transitions"),
                                      (axs[1], "Observed throughput", "Million transitions", "Transitions / second")):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if speed_found and len(runs) > 1:
            ax.legend(fontsize=8)
    if speed_found:
        save_figure(fig, outdir, "09_training_speed.png", plt)
        produced.append("09_training_speed.png")
    else:
        plt.close(fig)
    return produced


def create_trace_charts(traces: list[Trace], outdir: Path) -> list[str]:
    if not traces:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required. Install it with: pip install matplotlib") from exc

    produced = []
    axes = {
        "前進": ("vx_body", "cmd_vx", "Forward speed (m/s)"),
        "後退": ("vx_body", "cmd_vx", "Backward speed (m/s)"),
        "左移動": ("vy_body", "cmd_vy", "Left speed (m/s)"),
        "左旋回": ("wz_body", "cmd_wz", "Yaw rate (rad/s)"),
    }
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    any_data = False
    for ax, (task, (measured, commanded, title)) in zip(axs.flat, axes.items()):
        target_drawn = False
        for trace in traces:
            rows = sorted((r for r in trace.rows if r["task"] == task and r["active"] >= 0.5),
                          key=lambda r: r["time_s"])
            if not rows:
                continue
            any_data = True
            x = [r["time_s"] for r in rows]
            ax.plot(x, [r[measured] for r in rows], linewidth=1.6, label=trace.label)
            if not target_drawn:
                ax.plot(x, [r[commanded] for r in rows], color="black", linestyle="--",
                        linewidth=1.2, label="command")
                target_drawn = True
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        if target_drawn:
            ax.legend(fontsize=8)
    if any_data:
        save_figure(fig, outdir, "10_velocity_tracking_trace.png", plt)
        produced.append("10_velocity_tracking_trace.png")
    else:
        plt.close(fig)

    for field, title, filename, ylabel in (
        ("base_height", "Base height", "11_base_height_trace.png", "Height (m)"),
        ("reward_total", "Cumulative return", "12_cumulative_return_trace.png", "Return"),
    ):
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        any_data = False
        for ax, (task, label) in zip(axs.flat, TASKS.items()):
            for trace in traces:
                rows = sorted((r for r in trace.rows if r["task"] == task and r["active"] >= 0.5),
                              key=lambda r: r["time_s"])
                if not rows:
                    continue
                any_data = True
                x = [r["time_s"] for r in rows]
                y = [r[field] for r in rows]
                if field == "reward_total":
                    total = 0.0
                    y = [total := total + v for v in y]
                line, = ax.plot(x, y, linewidth=1.6, label=trace.label)
                if field == "base_height":
                    falls = [r for r in rows if r["fallen"] >= 0.5]
                    if falls:
                        ax.scatter([falls[0]["time_s"]], [falls[0]["base_height"]],
                                   marker="x", s=55, linewidths=1.7, color=line.get_color())
            ax.set_title(f"{label}: {title}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if len(traces) > 1:
                ax.legend(fontsize=8)
        if any_data:
            save_figure(fig, outdir, filename, plt)
            produced.append(filename)
        else:
            plt.close(fig)
    return produced


def export_csv(runs: list[Run], outdir: Path) -> None:
    with (outdir / "all_scalars.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "tag", "transitions", "millions", "value", "wall_time_unix"])
        for run in runs:
            for tag, values in sorted(run.series.items()):
                for point in values:
                    writer.writerow([run.label, tag, point.step, point.step / 1e6,
                                     point.value, point.wall_time])


def build_summary(runs: list[Run]) -> dict:
    summary = {}
    tags = (["train/episode_total", "train/episode_style", "train/episode_task",
             "train/episode_length", "test/min_survival", "test/min_tracking",
             "policy/max_kl", "policy/rejected_kl", "policy/lr",
             "train/decap_factor_last"] +
            [f"test/{task}" for task in TASKS] +
            [f"test/{task}_{metric}" for task in TASKS for metric in ("survival", "tracking")])
    for run in runs:
        all_steps = [p.step for values in run.series.values() for p in values]
        current = {}
        for tag in tags:
            values = run.get(tag)
            if values:
                current[tag] = {"last": values[-1].value,
                                "step": values[-1].step,
                                "best": max(p.value for p in values)}
        summary[run.label] = {"source": str(run.path),
                              "max_logged_transitions": max(all_steps, default=0),
                              "metrics": current}
    return summary


def write_readme(outdir: Path, runs: list[Run], traces: list[Trace],
                 files: list[str], summary: dict) -> None:
    lines = ["# APEX training report", "", "Learning graphs use TensorBoard scalar events; final rollout graphs use evaluation trace CSV files.",
             "Learning-graph x-axes show collected transitions.",
             "Smoothed curves are for readability; `all_scalars.csv` contains the original values.",
             "Prior OFF evaluation is the relevant measure of independent actor performance.", "",
             "## Runs", ""]
    for run in runs:
        lines.append(f"- **{run.label}**: `{run.path}`")
    if traces:
        lines += ["", "## Final evaluation traces", ""]
        lines += [f"- **{trace.label}**: `{trace.path}`" for trace in traces]
    lines += ["", "## Charts", ""]
    lines += [f"- [{name}]({name})" for name in files]
    lines += ["", "## Final logged values", "",
              "| Run | Transitions (M) | Style | Task | Min survival | Min tracking | Prior |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def latest(metrics: dict, tag: str) -> str:
        value = metrics.get(tag, {}).get("last")
        return "—" if value is None else f"{value:.3f}"

    for label, item in summary.items():
        m = item["metrics"]
        lines.append(f"| {label} | {item['max_logged_transitions'] / 1e6:.2f} | "
                     f"{latest(m, 'train/episode_style')} | {latest(m, 'train/episode_task')} | "
                     f"{latest(m, 'test/min_survival')} | {latest(m, 'test/min_tracking')} | "
                     f"{latest(m, 'train/decap_factor_last')} |")
    lines += ["", "A high return alone does not establish command tracking. Check survival and tracking for all four commands.",
              "Training episode values depend on the Action Prior during rollout. Evaluation curves always use Prior OFF.",
              "The reported speed includes evaluation, checkpoint saves, and any videos produced during training.", ""]
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"),
                        help="TensorBoard runs root; used when --run is omitted (default: runs)")
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="A specific run directory; repeat to compare runs")
    parser.add_argument("--trace", type=Path, action="append", default=[],
                        help="A final evaluation trace CSV; repeat to compare traces")
    parser.add_argument("--trace-root", type=Path,
                        help="Directory containing params_*_APEX folders with final evaluation traces")
    parser.add_argument("--label", action="append", default=[],
                        help="Display label for each --run; repeat in the same order")
    parser.add_argument("--match", default="go2_snn_apex",
                        help="Filter for auto-discovery (default: go2_snn_apex; empty means all)")
    parser.add_argument("--output", type=Path, default=Path("apex_report"),
                        help="Output directory (default: apex_report)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Number of logged points in trailing moving average (default: 5)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.smooth < 1:
        raise ValueError("--smooth must be >= 1")
    if args.label and len(args.label) != len(args.run):
        raise ValueError("Provide exactly one --label for each --run")
    if args.run:
        paths = [p.resolve() for p in args.run]
    elif args.runs_root.is_dir():
        paths = discover_runs(args.runs_root, args.match)
    else:
        paths = []
    if args.trace:
        trace_paths = [p.resolve() for p in args.trace]
    else:
        base = args.trace_root.resolve() if args.trace_root else args.runs_root.resolve().parent
        trace_paths = sorted(base.glob("params_*_APEX/*_eval_trace_*.csv")) if base.is_dir() else []
    if not paths and not trace_paths:
        raise FileNotFoundError("No TensorBoard logs or evaluation traces found. Check --runs-root, --run, or --trace.")
    labels = args.label if args.label else [normalize_label(p) for p in paths]
    counts = {}
    unique_labels = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        unique_labels.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    runs = []
    for path, label in zip(paths, unique_labels):
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {path}")
        run = load_run(path, label)
        print(f"Loaded {label}: {len(run.series)} scalar tags from {path}")
        runs.append(run)
    traces = [load_trace(path) for path in trace_paths]
    for trace in traces:
        print(f"Loaded evaluation trace {trace.label}: {len(trace.rows)} rows from {trace.path}")
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    export_csv(runs, outdir)
    summary = build_summary(runs)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files = create_charts(runs, outdir, args.smooth) if runs else []
    files += create_trace_charts(traces, outdir)
    write_readme(outdir, runs, traces, files, summary)
    print(f"Report: {outdir / 'README.md'}")
    print(f"Created {len(files)} charts, all_scalars.csv, and summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
