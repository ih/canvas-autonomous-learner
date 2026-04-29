"""Subprocess wrapper around canvas-world-model CLI tools.

Keeps the learner process light (no torch training graphs) and ensures a
training crash doesn't take down the state machine. The driver owns the
sequence: build canvases -> merge datasets -> train (optionally from
scratch) -> evaluate.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .gpu_monitor import GpuMonitor


# --------------------------------------------------- subprocess exceptions


class SubprocessMemoryAbort(Exception):
    """Raised when the GPU monitor reports sustained VRAM pressure during
    a training subprocess. The subprocess is killed before the
    exception is raised.
    """

    def __init__(self, tag: str, summary: dict, last_lines: Optional[list[str]] = None):
        self.tag = tag
        self.summary = summary or {}
        self.last_lines = last_lines or []
        super().__init__(f"[{tag}] memory abort: {self.summary}")


class SubprocessStalled(Exception):
    """Raised when a training subprocess emits no `training_progress`
    events for longer than `stall_timeout_s`. The most common cause on
    this host is VRAM saturation dropping throughput to near-zero.
    """

    def __init__(
        self,
        tag: str,
        seconds_since_last_progress: float,
        summary: Optional[dict] = None,
        last_lines: Optional[list[str]] = None,
    ):
        self.tag = tag
        self.seconds_since_last_progress = float(seconds_since_last_progress)
        self.summary = summary or {}
        self.last_lines = last_lines or []
        super().__init__(
            f"[{tag}] stalled: {self.seconds_since_last_progress:.1f}s "
            f"since last progress"
        )


class SubprocessTimeout(Exception):
    """Raised when a subprocess exceeds its hard wall-clock timeout."""

    def __init__(self, tag: str, timeout_s: float):
        self.tag = tag
        self.timeout_s = float(timeout_s)
        super().__init__(f"[{tag}] hard timeout after {self.timeout_s:.0f}s")


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ------------------------- stdout parsers for training progress -------

# `train_diffusion.py` prints `Train: N canvases, Val: M canvases` at startup.
_TRAIN_SIZE_RE = re.compile(
    r"^Train:\s*(\d+)\s+canvases,\s*Val:\s*(\d+)\s+canvases"
)

# Per-epoch summary lines look like:
#   "Epoch 12/300: train_loss=0.123456, val_loss=0.234567, lr=3.00e-04, ..."
_EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)/(\d+):\s*train_loss=([-\d.eE]+),\s*val_loss=([-\d.eE]+)"
    r"(?:,\s*lr=([-\d.eE]+))?"
    r"(?:,\s*best_val=([-\d.eE]+))?"
)


def _emit_training_line_events(tag: str, line: str, event_log) -> None:
    """If `tag` is a training subprocess, parse `line` and emit structured
    events (`training_dataset_size`, `training_progress`). Otherwise no-op.
    """
    if tag != "train_diffusion" or event_log is None:
        return
    m = _TRAIN_SIZE_RE.search(line)
    if m:
        event_log.log(
            "training_dataset_size",
            train_canvases=int(m.group(1)),
            val_canvases=int(m.group(2)),
        )
        return
    m = _EPOCH_RE.search(line)
    if m:
        event_log.log(
            "training_progress",
            epoch=int(m.group(1)),
            total_epochs=int(m.group(2)),
            train_loss=float(m.group(3)),
            val_loss=float(m.group(4)),
            lr=float(m.group(5)) if m.group(5) else None,
            best_val=float(m.group(6)) if m.group(6) else None,
        )


def _kill_subprocess(proc: subprocess.Popen) -> None:
    """Terminate then kill a subprocess, best-effort."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    except Exception:
        pass


def _run(
    cmd: list[str],
    cwd: str | Path | None = None,
    event_log=None,
    tag: str = "",
    *,
    monitor: Optional[GpuMonitor] = None,
    abort_frac: float = 0.93,
    stall_timeout_s: Optional[float] = None,
    hard_timeout_s: Optional[float] = None,
) -> None:
    """Spawn a subprocess, stream stdout line-by-line, and (for tagged
    training invocations) parse per-epoch progress into `training_progress`
    events so the live dashboard can show a training loss chart while the
    subprocess is still running.

    Non-training subprocesses (create_dataset, combine_datasets, evaluate)
    still stream stdout to this process's stdout so the operator can tail
    them from the terminal, but nothing is parsed from their output.

    When `monitor` is passed, poll it for sustained VRAM pressure and
    raise `SubprocessMemoryAbort` if the threshold is breached. When
    `stall_timeout_s` is set (training only), raise `SubprocessStalled`
    if no `training_progress` line arrives within that window. When
    `hard_timeout_s` is set, raise `SubprocessTimeout` after that total
    wall-clock duration.
    """
    if event_log is not None:
        event_log.log("subprocess_start", tag=tag, cmd=cmd, cwd=str(cwd) if cwd else None)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line buffered
        )
    except (OSError, FileNotFoundError) as e:
        if event_log is not None:
            event_log.log("subprocess_failed", tag=tag, error=str(e))
        raise

    # Reader thread pushes each stdout line onto a queue; a `None` sentinel
    # marks end-of-stream so the main loop knows when to stop polling.
    line_q: queue.Queue = queue.Queue()

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line_q.put(raw.rstrip("\r\n"))
        finally:
            line_q.put(None)

    reader = threading.Thread(target=_reader, name=f"_run-{tag}", daemon=True)
    reader.start()

    rolling_lines: deque[str] = deque(maxlen=30)
    start_t = time.time()
    last_progress_t = start_t
    poll_tick_s = 2.0

    try:
        while True:
            end_of_stream = False
            try:
                line = line_q.get(timeout=poll_tick_s)
            except queue.Empty:
                line = "__TICK__"

            if line is None:
                end_of_stream = True
            elif line == "__TICK__":
                pass
            else:
                rolling_lines.append(line)
                if line:
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                    if tag == "train_diffusion" and _EPOCH_RE.search(line):
                        last_progress_t = time.time()
                    _emit_training_line_events(tag, line, event_log)

            # Abort conditions — checked every tick regardless of whether a
            # line arrived, so we can intercept a subprocess that has gone
            # silent from VRAM pressure.
            if monitor is not None and monitor.is_under_pressure(abort_frac):
                summary = monitor.summary()
                last_n = list(rolling_lines)
                if event_log is not None:
                    event_log.log(
                        "training_memory_abort",
                        tag=tag,
                        summary=summary,
                        last_lines=last_n,
                    )
                _kill_subprocess(proc)
                raise SubprocessMemoryAbort(tag, summary, last_n)

            if stall_timeout_s and tag == "train_diffusion":
                gap = time.time() - last_progress_t
                if gap > float(stall_timeout_s):
                    summary = monitor.summary() if monitor is not None else {}
                    last_n = list(rolling_lines)
                    if event_log is not None:
                        event_log.log(
                            "training_stalled",
                            tag=tag,
                            seconds_since_last_progress=gap,
                            stall_timeout_s=float(stall_timeout_s),
                            summary=summary,
                            last_lines=last_n,
                        )
                    _kill_subprocess(proc)
                    raise SubprocessStalled(tag, gap, summary, last_n)

            if hard_timeout_s:
                total = time.time() - start_t
                if total > float(hard_timeout_s):
                    if event_log is not None:
                        event_log.log(
                            "subprocess_timeout",
                            tag=tag,
                            timeout_s=float(hard_timeout_s),
                        )
                    _kill_subprocess(proc)
                    raise SubprocessTimeout(tag, float(hard_timeout_s))

            if end_of_stream:
                break
    finally:
        # Make sure the reader thread is not left blocked on proc.stdout.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_subprocess(proc)
        reader.join(timeout=5)

    if proc.returncode != 0:
        if event_log is not None:
            event_log.log("subprocess_failed", tag=tag, returncode=proc.returncode)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    if event_log is not None:
        event_log.log("subprocess_done", tag=tag)


def _read_eval_visual_mse(eval_output_dir: Path) -> Optional[float]:
    report = eval_output_dir / "report.json"
    if not report.exists():
        return None
    with open(report) as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    val = metrics.get("val_mse_visual")
    if val is None:
        val = metrics.get("val_mse")
    return float(val) if val is not None else None


def _read_per_cell_mse(eval_output_dir: Path) -> Optional[dict]:
    """Read the per-(joint, position-bin) MSE breakdown from an eval run.

    Returns the dict produced by `evaluate.py`'s `per_cell_mse` field, or
    None if the report is missing or the field isn't present (older
    canvases predate sub-phase 1's metadata; older eval runs predate
    sub-phase 2's per-cell accumulator). Consumed by the per-joint
    sub-burst planner.
    """
    report = eval_output_dir / "report.json"
    if not report.exists():
        return None
    with open(report) as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    cells = metrics.get("per_cell_mse")
    return cells if isinstance(cells, dict) else None


def _motor_bounds_arg(cfg) -> list[str]:
    """Return `[--motor-bounds-json, <json>]` if cfg.training.motor_bounds
    is set, else `[]`. Shared by both `build_canvases` and `combine_datasets`
    so every offline canvas step uses identical normalization bounds.
    """
    training = getattr(cfg, "training", None)
    mb = getattr(training, "motor_bounds", None) if training is not None else None
    if mb is None:
        return []
    # SimpleNamespace or dict — normalize to dict for JSON
    if hasattr(mb, "__dict__"):
        mb_dict = {k: list(v) for k, v in vars(mb).items() if v is not None}
    else:
        mb_dict = {k: list(v) for k, v in dict(mb).items() if v is not None}
    if not mb_dict:
        return []
    return ["--motor-bounds-json", json.dumps(mb_dict)]


def build_canvases(
    cfg,
    new_lerobot_dir: str | Path,
    output_dir: str | Path,
    event_log=None,
) -> Path:
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    output_dir = Path(output_dir)
    cmd = [
        python_exe,
        "create_dataset.py",
        "--lerobot-path", str(new_lerobot_dir),
        "--output", str(output_dir),
    ]
    cmd.extend(_motor_bounds_arg(cfg))
    _run(
        cmd,
        cwd=cwm,
        event_log=event_log,
        tag="create_dataset",
    )
    return output_dir


def combine_datasets(
    cfg,
    inputs: list[str | Path],
    output_dir: str | Path,
    event_log=None,
) -> Path:
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    cmd = [python_exe, "combine_datasets.py", "--inputs"]
    cmd.extend(str(p) for p in inputs)
    cmd.extend(["--output", str(output_dir)])
    cmd.extend(_motor_bounds_arg(cfg))
    _run(cmd, cwd=cwm, event_log=event_log, tag="combine_datasets")
    return Path(output_dir)


_TRAINING_FLAG_MAP: list[tuple[str, str]] = [
    # (cfg.training.<attr>, train_diffusion.py CLI flag)
    ("patch_size", "--patch-size"),
    ("embed_dim", "--embed-dim"),
    ("depth", "--depth"),
    ("num_heads", "--num-heads"),
    ("num_train_timesteps", "--num-train-timesteps"),
    ("beta_schedule", "--beta-schedule"),
    ("prediction_type", "--prediction-type"),
    ("lr", "--lr"),
    ("weight_decay", "--weight-decay"),
    ("lr_schedule", "--lr-schedule"),
    ("warmup_epochs", "--warmup-epochs"),
    ("min_lr", "--min-lr"),
    ("grad_clip", "--grad-clip"),
    ("batch_size", "--batch-size"),
    ("gradient_accumulation_steps", "--gradient-accumulation-steps"),
    ("seed", "--seed"),
    ("val_ratio", "--val-ratio"),
]

# Boolean training flags forwarded as bare --flag (no value) when truthy.
# Used for the bf16 / grad-checkpointing / 8-bit-Adam mechanics that scale
# trainable model size on a single 32GB card.
_TRAINING_BOOL_FLAGS: list[tuple[str, str]] = [
    ("bf16", "--bf16"),
    ("gradient_checkpointing", "--gradient-checkpointing"),
    ("use_8bit_adam", "--use-8bit-adam"),
]


def _forward_training_hparams(cfg, cmd: list[str]) -> None:
    """Append `--flag value` for every attribute present in `cfg.training`.

    Missing attributes / missing `cfg.training` → flag is omitted and
    train_diffusion.py's own default applies. This keeps the continuous-loop
    path backward compatible — configs without a `training:` block behave
    exactly as before.
    """
    training = getattr(cfg, "training", None)
    if training is None:
        return
    for attr, flag in _TRAINING_FLAG_MAP:
        value = getattr(training, attr, None)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    for attr, flag in _TRAINING_BOOL_FLAGS:
        if bool(getattr(training, attr, False)):
            cmd.append(flag)


def train(
    cfg,
    dataset: str | Path,
    checkpoint_dir: str | Path,
    epochs: int,
    resume_checkpoint: str | Path | None = None,
    event_log=None,
) -> Path:
    """Run `train_diffusion.py`.

    - `resume_checkpoint=None` → cold start (fresh model, fresh optimizer).
    - `resume_checkpoint` set → `--fine-tune <path>`: loads **weights only**,
      starts a fresh optimizer and cosine scheduler so the LR curve restarts
      from `--lr` instead of inheriting the dead end-of-previous-cycle LR.
      (The alternative flag `--resume` restores optimizer/scheduler state
      too, which is wrong for cumulative retraining — it freezes the LR at
      the prior cycle's terminal value.)

    `epochs` is always the absolute number of epochs this invocation runs,
    since `--fine-tune` resets the epoch counter to 0. In practice it acts
    as a ceiling because `--early-stop-patience` (read from
    `cfg.cadence.early_stop_patience`) will stop earlier once val plateaus.

    Architectural + optimization hyperparameters are forwarded from
    `cfg.training.*` when present. The comparison experiment sets these to
    match Arm A's offline training protocol exactly.
    """
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        "train_diffusion.py",
        "--dataset", str(dataset),
        "--epochs", str(int(epochs)),
        "--checkpoint-dir", str(checkpoint_dir),
        "--no-wandb",
    ]
    _forward_training_hparams(cfg, cmd)

    early_stop = int(getattr(cfg.cadence, "early_stop_patience", 0) or 0)
    if early_stop > 0:
        cmd.extend(["--early-stop-patience", str(early_stop)])

    if resume_checkpoint is not None:
        cmd.extend(["--fine-tune", str(resume_checkpoint)])
        if event_log is not None:
            event_log.log(
                "train_fine_tune",
                from_checkpoint=str(resume_checkpoint),
                epochs=int(epochs),
                early_stop_patience=early_stop,
            )

    # GPU monitor: poll nvidia-smi, abort on sustained VRAM pressure or
    # on training_progress stall. Thresholds are read from cfg.gpu with
    # conservative defaults tuned for a 32 GB RTX 5090.
    gpu_cfg = getattr(cfg, "gpu", None)
    abort_frac = float(getattr(gpu_cfg, "memory_abort_frac", 0.93)) if gpu_cfg else 0.93
    warn_frac = float(getattr(gpu_cfg, "memory_warn_frac", 0.85)) if gpu_cfg else 0.85
    sample_interval_s = float(getattr(gpu_cfg, "sample_interval_s", 5.0)) if gpu_cfg else 5.0
    stall_timeout_s = float(getattr(cfg.cadence, "training_stall_timeout_s", 600.0))
    hard_timeout_s = float(getattr(cfg.cadence, "retrain_timeout_s", 7200.0))

    monitor = GpuMonitor(
        sample_interval_s=sample_interval_s,
        warn_frac=warn_frac,
        abort_frac=abort_frac,
        event_log=event_log,
    )
    monitor.start()
    try:
        _run(
            cmd,
            cwd=cwm,
            event_log=event_log,
            tag="train_diffusion",
            monitor=monitor,
            abort_frac=abort_frac,
            stall_timeout_s=stall_timeout_s,
            hard_timeout_s=hard_timeout_s,
        )
    finally:
        monitor.stop()

    # train_diffusion.py writes `best.pth` (only when val improves) and
    # `final.pth` (always, at the end). Prefer best.pth; fall back to
    # final.pth when the fine-tune didn't beat its own starting val.
    best = checkpoint_dir / "best.pth"
    final = checkpoint_dir / "final.pth"
    if best.exists():
        return best
    if final.exists():
        if event_log is not None:
            event_log.log("train_used_final_pth", reason="no best.pth improvement")
        return final
    raise FileNotFoundError(f"no checkpoint written under {checkpoint_dir}")


# Backward-compat alias — the old name is still imported by existing tests /
# scripts. New code should call `train()` directly.
def fine_tune(
    cfg,
    merged_dataset: str | Path,
    resume_ckpt: str | Path,
    checkpoint_dir: str | Path,
    epochs: int,
    event_log=None,
) -> Path:
    return train(
        cfg, merged_dataset, checkpoint_dir, epochs,
        resume_checkpoint=resume_ckpt, event_log=event_log,
    )


def evaluate(
    cfg,
    checkpoint: str | Path,
    dataset: str | Path,
    output_dir: str | Path,
    event_log=None,
) -> Optional[float]:
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            python_exe,
            "evaluate.py",
            "--model-type", "diffusion",
            "--checkpoint", str(checkpoint),
            "--dataset", str(dataset),
            "--output-dir", str(output_dir),
            "--no-html",
        ],
        cwd=cwm,
        event_log=event_log,
        tag="evaluate",
    )
    return _read_eval_visual_mse(output_dir)


def retrain_cumulative(
    cfg,
    accumulated_canvas_dirs: list[str | Path],
    resume_checkpoint: str | Path | None,
    epochs: int,
    locked_val_dataset: str | Path | None = None,
    event_log=None,
) -> dict | None:
    """Cumulative retrain used by the comparison experiment.

    Merges *all* accumulated canvas dirs into one dataset, then trains with
    or without `--resume`. Evaluates on both the merged training dataset (for
    `train_val_mse` guard signal) and optionally on a locked val set (for the
    cross-arm comparison metric). This function does NOT promote checkpoints
    — promotion / rejection is the orchestrator's job.

    Returns `{checkpoint, merged_dataset, train_val_mse, locked_val_mse}` on
    success, or `None` on any subprocess failure.
    """
    stamp = _stamp()
    canvas_out = Path(cfg.paths.canvas_out)
    ckpt_dir = Path(cfg.paths.ckpt_dir)

    merged_dir = canvas_out / f"merged_{stamp}"
    ft_ckpt_dir = ckpt_dir / f"ft_{stamp}"
    eval_out_train = Path(cfg.paths.runs_dir) / f"eval_train_{stamp}"
    eval_out_locked = Path(cfg.paths.runs_dir) / f"eval_locked_{stamp}"

    try:
        dirs = [Path(d) for d in accumulated_canvas_dirs]
        if not dirs:
            raise ValueError("accumulated_canvas_dirs is empty")

        if len(dirs) == 1:
            # combine_datasets would be a no-op; copy the single dir so the
            # merged path is stable and the caller can still read from it.
            shutil.copytree(dirs[0], merged_dir)
        else:
            combine_datasets(cfg, dirs, merged_dir, event_log=event_log)

        new_ckpt = train(
            cfg,
            merged_dir,
            ft_ckpt_dir,
            epochs=epochs,
            resume_checkpoint=resume_checkpoint,
            event_log=event_log,
        )

        train_val_mse = evaluate(
            cfg, new_ckpt, merged_dir, eval_out_train, event_log=event_log
        )
        # Per-(joint, position-bin) MSE breakdown from the merged-train
        # eval — the same one that produced train_val_mse. Consumed by
        # the per-joint sub-burst planner when cadence.per_joint_targeting
        # is enabled. None for older datasets/checkpoints.
        per_cell_mse = _read_per_cell_mse(eval_out_train)

        locked_val_mse = None
        if locked_val_dataset is not None and Path(locked_val_dataset).exists():
            locked_val_mse = evaluate(
                cfg, new_ckpt, locked_val_dataset, eval_out_locked,
                event_log=event_log,
            )
        elif locked_val_dataset is not None and event_log is not None:
            event_log.log(
                "locked_val_skipped",
                reason="dataset path not found",
                path=str(locked_val_dataset),
            )

        # Per-joint locked-val evaluations — optional. When the config
        # carries locked_val_shoulder / locked_val_elbow paths to
        # separate held-out corpora, evaluate each so the learner can
        # track joint-specific learning curves for the two-joint
        # experiment. The advisor does NOT see these values (real-run
        # fidelity — a production run has no locked val), but they're
        # logged to the registry + events for post-hoc analysis.
        # Missing paths are silently skipped so the experiment can start
        # before the val corpus has been recorded.
        locked_val_shoulder = None
        locked_val_elbow = None
        shoulder_path = getattr(cfg.paths, "locked_val_shoulder", None)
        elbow_path = getattr(cfg.paths, "locked_val_elbow", None)
        if shoulder_path and Path(shoulder_path).exists():
            locked_val_shoulder = evaluate(
                cfg, new_ckpt, shoulder_path,
                Path(cfg.paths.runs_dir) / f"eval_locked_shoulder_{stamp}",
                event_log=event_log,
            )
        if elbow_path and Path(elbow_path).exists():
            locked_val_elbow = evaluate(
                cfg, new_ckpt, elbow_path,
                Path(cfg.paths.runs_dir) / f"eval_locked_elbow_{stamp}",
                event_log=event_log,
            )
    except SubprocessMemoryAbort as e:
        return {
            "memory_abort": True,
            "tag": e.tag,
            "summary": e.summary,
            "last_lines": e.last_lines,
        }
    except SubprocessStalled as e:
        return {
            "stalled": True,
            "tag": e.tag,
            "seconds_since_last_progress": e.seconds_since_last_progress,
            "summary": e.summary,
            "last_lines": e.last_lines,
        }
    except SubprocessTimeout as e:
        return {
            "timeout": True,
            "tag": e.tag,
            "timeout_s": e.timeout_s,
        }
    except Exception as e:
        if event_log is not None:
            event_log.log("retrain_exception", error=str(e))
        return None

    if train_val_mse is None:
        if event_log is not None:
            event_log.log("retrain_no_train_val_mse")
        return None

    if event_log is not None:
        event_log.log(
            "retrain_done",
            checkpoint=str(new_ckpt),
            merged_dataset=str(merged_dir),
            train_val_mse=train_val_mse,
            locked_val_mse=locked_val_mse,
            locked_val_shoulder=locked_val_shoulder,
            locked_val_elbow=locked_val_elbow,
            per_cell_mse_joints=(
                sorted(per_cell_mse.keys()) if per_cell_mse else None
            ),
            num_canvas_dirs=len(dirs),
            epochs=epochs,
            from_scratch=(resume_checkpoint is None),
        )
    return {
        "checkpoint": str(new_ckpt),
        "merged_dataset": str(merged_dir),
        "train_val_mse": train_val_mse,
        "locked_val_mse": locked_val_mse,
        "locked_val_shoulder": locked_val_shoulder,
        "locked_val_elbow": locked_val_elbow,
        "per_cell_mse": per_cell_mse,
    }


def retrain(
    cfg,
    new_lerobot_dir: str | Path,
    live_checkpoint: str,
    base_canvas_dataset: str,
    val_dataset: str,
    baseline_val_mse: float | None,
    event_log=None,
) -> dict | None:
    """Legacy retrain path used by the original continuous loop.

    Kept for backward compatibility with the existing `main_loop` and its
    tests. For the comparison experiment, use `retrain_cumulative` instead.
    """
    stamp = _stamp()
    canvas_out = Path(cfg.paths.canvas_out)
    ckpt_dir = Path(cfg.paths.ckpt_dir)

    new_canvas_dir = canvas_out / f"new_batch_{stamp}"
    merged_dir = canvas_out / f"merged_{stamp}"
    ft_ckpt_dir = ckpt_dir / f"ft_{stamp}"
    eval_out = Path(cfg.paths.runs_dir) / f"eval_{stamp}"

    try:
        build_canvases(cfg, new_lerobot_dir, new_canvas_dir, event_log=event_log)
        combine_datasets(
            cfg,
            [base_canvas_dataset, new_canvas_dir],
            merged_dir,
            event_log=event_log,
        )
        new_ckpt = fine_tune(
            cfg,
            merged_dir,
            live_checkpoint,
            ft_ckpt_dir,
            epochs=cfg.cadence.ft_epochs,
            event_log=event_log,
        )
        val_mse = evaluate(cfg, new_ckpt, val_dataset, eval_out, event_log=event_log)
    except Exception as e:
        if event_log is not None:
            event_log.log("retrain_exception", error=str(e))
        return None

    if val_mse is None:
        if event_log is not None:
            event_log.log("retrain_no_val_mse")
        return None

    guard = cfg.thresholds.val_guard
    if baseline_val_mse is not None and val_mse > guard * baseline_val_mse:
        if event_log is not None:
            event_log.log(
                "retrain_rejected",
                val_mse=val_mse,
                baseline=baseline_val_mse,
                guard=guard,
            )
        return None

    if event_log is not None:
        event_log.log(
            "retrain_accepted",
            checkpoint=str(new_ckpt),
            merged_dataset=str(merged_dir),
            val_mse=val_mse,
        )
    return {
        "checkpoint": str(new_ckpt),
        "merged_dataset": str(merged_dir),
        "val_mse": val_mse,
    }
