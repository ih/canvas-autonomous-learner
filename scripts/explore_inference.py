"""Replay-inference probe worker for the comparison experiment.

Tails the current EXPLORE phase's LeRobot dataset as episodes get written,
reuses canvas-world-model's `data.lerobot_loader.load_episode` to extract
correctly-aligned `(before_frame, action, after_frame, motor_state)` tuples
using the discrete action log's frame indices, runs the current live
checkpoint on each, and writes a `before | predicted | actual` probe grid
PNG to the current session's `examples_*/` dir. The live dashboard picks
these up automatically via its existing file-watcher.

Why this exists:
- The live-camera probes in `learner.verifier` suffer from DShow buffer
  staleness — the "after" frame often catches mid-motion because
  `videocapture.grab()` on Windows doesn't guarantee a post-action frame.
- The offline canvas-building pipeline already solves this by recording
  full LeRobot episodes at 10 fps and picking `(frame_idx, frame_idx +
  action_duration * fps)` pairs from the discrete action log — guaranteed
  correct alignment by construction.
- This worker reuses that exact mechanism for *live* dashboard probes.
  No hardware contention (the explorer subprocess is the only thing
  touching the robot), no camera buffer fix needed, and alignment is
  identical to the training data the canvas model actually sees.

Run it in a separate terminal from the learner:
    python scripts/explore_inference.py --config configs/comparison_armB.yaml

It detects EXPLORE phases from `runs/events_*.jsonl`, activates when
EXPLORE is running, and releases the GPU during retrain so it doesn't
contend with `train_diffusion.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner.events import EventLog  # noqa: E402
from learner.action_canvas import save_action_canvas  # noqa: E402
from learner.registry import Registry  # noqa: E402


# ---------------------------------------------------------------- sys.path

def _ensure_sibling_paths(cfg) -> None:
    """Add sibling repo roots so we can import their modules."""
    for attr in ("canvas_world_model", "canvas_robot_control"):
        p = getattr(cfg.paths, attr, None)
        if p and p not in sys.path:
            sys.path.insert(0, p)


# ----------------------------------------------------------- phase tracker

class PhaseTracker:
    """Tail the session's events.jsonl and expose the current phase.

    Phases of interest:
      - "explore"   : explorer subprocess is recording episodes (active work)
      - "retrain"   : train_diffusion subprocess running (pause to free GPU)
      - "idle"      : between phases, short
      - "done"      : experiment_done seen, exit
      - "unknown"   : no events yet / no session yet
    """

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.session: Optional[str] = None
        self.events_path: Optional[Path] = None
        self.read_pos: int = 0
        self.phase: str = "unknown"
        self.explore_repo_id: Optional[str] = None
        self.current_cycle: int = 0

    def _latest_session(self) -> Optional[str]:
        cands = list(self.runs_dir.glob("events_*.jsonl"))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime).stem.removeprefix("events_")

    def update(self) -> None:
        session = self._latest_session()
        if session is None:
            self.phase = "unknown"
            return
        if session != self.session:
            # New session — reset state.
            self.session = session
            self.events_path = self.runs_dir / f"events_{session}.jsonl"
            self.read_pos = 0
            self.phase = "unknown"
            self.explore_repo_id = None
            self.current_cycle = 0

        if not self.events_path or not self.events_path.exists():
            return

        try:
            with open(self.events_path, "r", encoding="utf-8") as f:
                f.seek(self.read_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._apply(ev)
                self.read_pos = f.tell()
        except OSError:
            pass  # file may be mid-write, try again next poll

    def _apply(self, ev: dict) -> None:
        name = ev.get("event")
        if name == "cycle_start":
            self.current_cycle = int(ev.get("cycle", 0))
        elif name == "explore_start":
            self.phase = "explore"
            self.explore_repo_id = ev.get("repo_id")
        elif name == "explore_done":
            self.phase = "idle"
        elif name == "explore_failed":
            self.phase = "idle"
        elif name == "retrain_start":
            self.phase = "retrain"
        elif name == "retrain_done":
            self.phase = "idle"
        elif name == "experiment_done":
            self.phase = "done"
        elif name == "shutdown":
            self.phase = "done"


# ------------------------------------------------- predictor (lazy load)

class PredictorState:
    """Holds a `WorldModelPredictor` bound to the current live checkpoint.

    Loads lazily when EXPLORE becomes active. Releases VRAM when the
    phase transitions away from EXPLORE (e.g., retrain about to run)
    so `train_diffusion.py` has the GPU to itself.
    """

    def __init__(self, cfg, registry: Registry):
        self.cfg = cfg
        self.registry = registry
        self.predictor = None
        self.loaded_checkpoint: Optional[str] = None

    def ensure_loaded(self) -> bool:
        """Return True if a predictor is ready; False if no live checkpoint yet."""
        ckpt = self.registry.live_checkpoint()
        if ckpt is None:
            return False
        if ckpt != self.loaded_checkpoint:
            self.release()
            _ensure_sibling_paths(self.cfg)
            from control.world_model import WorldModelPredictor  # type: ignore
            self.predictor = WorldModelPredictor(
                checkpoint_path=ckpt,
                canvas_world_model_path=self.cfg.paths.canvas_world_model,
            )
            self.predictor.load()
            self.loaded_checkpoint = ckpt
            print(f"[worker] loaded checkpoint: {ckpt}")
        return True

    def release(self) -> None:
        if self.predictor is None:
            return
        del self.predictor
        self.predictor = None
        self.loaded_checkpoint = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("[worker] released predictor (GPU freed)")


# --------------------------------------------------- episode processing

def _lerobot_cache_dir_for_repo(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _list_completed_episode_indices(cache_dir: Path) -> list[int]:
    """Return sorted episode indices that have a completed discrete action log.

    LeRobot's data chunks are file-000.parquet per chunk (many episodes
    concatenated), not one file per episode, so we can't list episodes by
    parquet filename. The discrete_action_logs directory, however, does
    write one `episode_NNNNNN.jsonl` per episode, and the file is only
    flushed when the episode completes — so its presence is both the
    listing source AND the completeness signal.
    """
    log_dir = cache_dir / "meta" / "discrete_action_logs"
    if not log_dir.exists():
        return []
    indices: list[int] = []
    for jl in log_dir.glob("episode_*.jsonl"):
        try:
            indices.append(int(jl.stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return sorted(indices)


def _parquet_chunk_stable(cache_dir: Path, quiescent_seconds: float = 1.0) -> bool:
    """True iff every parquet chunk under `cache_dir` hasn't been touched
    for at least `quiescent_seconds`. Used to avoid reading mid-flush files
    whose footer ("Parquet magic bytes") isn't written yet.
    """
    data_dir = cache_dir / "data"
    if not data_dir.exists():
        return True
    now = time.time()
    for p in data_dir.rglob("*.parquet"):
        try:
            if now - p.stat().st_mtime < quiescent_seconds:
                return False
        except OSError:
            return False
    return True


def _canvas_control_joint_idx(cfg) -> int:
    """Match WorldModelPredictor.predict_batch's control_joint_idx semantics."""
    _ensure_sibling_paths(cfg)
    from control.robot_interface import JOINTS  # type: ignore
    return JOINTS.index(cfg.robot.control_joint)


def process_episode(
    cfg,
    cache_dir: Path,
    episode_index: int,
    predictor_state: PredictorState,
    examples_dir: Path,
    cycle: int,
    event_log: EventLog,
) -> int:
    """Load one episode, run inference on each action, write probe PNGs.

    Returns the number of probes produced.
    """
    _ensure_sibling_paths(cfg)
    from data.lerobot_loader import LeRobotV3Reader, load_episode  # type: ignore
    from control.canvas_utils import FRAME_SIZE  # type: ignore

    reader = LeRobotV3Reader(str(cache_dir))
    cameras = [
        f"observation.images.{cfg.explore.base_camera_name}",
        f"observation.images.{cfg.explore.wrist_camera_name}",
    ]
    # canvas-world-model's loader strips the "observation.images." prefix
    # itself; it accepts just the short name. Build both forms — whichever
    # the video directory layout uses will resolve.
    short_cameras = [cfg.explore.base_camera_name, cfg.explore.wrist_camera_name]

    episode = None
    last_err = None
    for cam_list in (short_cameras, cameras):
        try:
            episode = load_episode(
                reader=reader,
                episode_index=episode_index,
                cameras=cam_list,
                stack_mode="vertical",
                frame_size=FRAME_SIZE,
                state_column="observation.state",
            )
            break
        except Exception as e:
            last_err = e
            continue
    if episode is None:
        event_log.log(
            "worker_episode_load_failed",
            episode_index=episode_index,
            error=str(last_err),
        )
        return 0

    if len(episode.actions) == 0 or len(episode.frames) < 2:
        return 0

    control_idx = _canvas_control_joint_idx(cfg)
    step_size = float(cfg.robot.step_size)

    predictor = predictor_state.predictor
    n_written = 0
    for i, action in enumerate(episode.actions):
        context_frame = episode.frames[i]        # stacked base+wrist
        actual_frame = episode.frames[i + 1]     # stacked base+wrist
        motor_state = (
            episode.motor_positions[i] if episode.motor_positions else None
        )
        if motor_state is None:
            continue

        try:
            pred_list = predictor.predict_batch(
                context_frame,
                motor_state,
                [int(action)],
                step_size=step_size,
                control_joint_idx=control_idx,
                prediction_depth=1,
            )
        except Exception as e:
            event_log.log(
                "worker_inference_failed",
                episode_index=episode_index,
                decision_index=i,
                error=str(e),
            )
            continue
        pred_base, pred_wrist = pred_list[0]

        # Split the vertically-stacked context/actual into per-camera views.
        h = actual_frame.shape[0] // 2
        before_base = context_frame[:h]
        before_wrist = context_frame[h:]
        actual_base = actual_frame[:h]
        actual_wrist = actual_frame[h:]

        import numpy as np
        pb = pred_base.astype(np.float32) / 255.0
        pw = pred_wrist.astype(np.float32) / 255.0
        ab = actual_base.astype(np.float32) / 255.0
        aw = actual_wrist.astype(np.float32) / 255.0
        mse = float(((pb - ab) ** 2).mean() + ((pw - aw) ** 2).mean()) / 2.0

        tag = f"c{cycle:03d}_ep{episode_index:04d}_d{i:02d}_{time.strftime('%H%M%S')}"
        out_path = examples_dir / f"action_canvas_{tag}.png"
        save_action_canvas(
            out_path,
            cams_before={"base": before_base, "wrist": before_wrist},
            pred_base=pred_base,
            pred_wrist=pred_wrist,
            actual_base=actual_base,
            actual_wrist=actual_wrist,
            mse=mse,
            action=int(action),
            header_prefix="[worker]",
        )
        n_written += 1
        event_log.log(
            "worker_probe",
            cycle=cycle,
            episode_index=episode_index,
            decision_index=i,
            action=int(action),
            mse=mse,
            source="lerobot_replay",
        )

    return n_written


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "comparison_armB.yaml"))
    p.add_argument("--poll-seconds", type=float, default=2.0,
                   help="How often to poll for new events + episodes.")
    args = p.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg.paths.runs_dir)
    registry = Registry(cfg.paths.registry_file)
    tracker = PhaseTracker(runs_dir)
    predictor_state = PredictorState(cfg, registry)

    # Separate event log so the learner's events aren't polluted with
    # worker events on the same file (we can still cross-reference by
    # timestamp).
    worker_log = EventLog(runs_dir, session="explore_inference_worker")
    print(f"[worker] events: {worker_log.path}")
    print(f"[worker] polling {runs_dir} every {args.poll_seconds}s")

    processed: dict[str, set[int]] = {}  # repo_id -> set of processed episode indices

    try:
        while True:
            tracker.update()

            if tracker.phase == "done":
                print("[worker] experiment_done seen; exiting")
                break

            if tracker.phase != "explore":
                # Release GPU during non-explore phases.
                if predictor_state.predictor is not None:
                    predictor_state.release()
                time.sleep(args.poll_seconds)
                continue

            if tracker.explore_repo_id is None:
                time.sleep(args.poll_seconds)
                continue

            cache_dir = _lerobot_cache_dir_for_repo(tracker.explore_repo_id)
            if not cache_dir.exists():
                time.sleep(args.poll_seconds)
                continue

            if not predictor_state.ensure_loaded():
                # No live checkpoint yet (cycle 0 hasn't finished retraining).
                time.sleep(args.poll_seconds)
                continue

            examples_dir = runs_dir / f"examples_{tracker.session}"
            processed_set = processed.setdefault(tracker.explore_repo_id, set())

            # Find newly-completed episodes via the discrete action log.
            # LeRobot writes one .jsonl per episode at episode-end, so its
            # presence implies the episode is done. We still skip the
            # newest-indexed one for a beat in case the parquet chunk
            # hasn't been updated yet (the .jsonl can be flushed a bit
            # ahead of the data chunk).
            indices = _list_completed_episode_indices(cache_dir)
            if not indices:
                time.sleep(args.poll_seconds)
                continue
            for idx in indices[:-1]:
                if idx in processed_set:
                    continue
                # Wait for the parquet chunk to settle before reading.
                # LeRobot buffers rows and only flushes the footer at chunk
                # boundaries, so a file that grew <1s ago might still be
                # missing its magic bytes. Skip this tick if the chunk is
                # still being written; try again on the next poll.
                if not _parquet_chunk_stable(cache_dir, quiescent_seconds=1.0):
                    break
                try:
                    n = process_episode(
                        cfg,
                        cache_dir,
                        idx,
                        predictor_state,
                        examples_dir,
                        tracker.current_cycle,
                        worker_log,
                    )
                    processed_set.add(idx)
                    if n:
                        print(f"[worker] cycle {tracker.current_cycle} episode {idx}: {n} probes")
                except Exception as e:
                    msg = str(e)
                    if "Parquet magic bytes" in msg or "magic bytes not found" in msg:
                        # Race: chunk still being flushed. Don't mark as
                        # processed — let the next tick retry.
                        break
                    worker_log.log(
                        "worker_episode_error",
                        episode_index=idx,
                        error=msg,
                        trace=traceback.format_exc().splitlines()[-3:],
                    )
                    processed_set.add(idx)  # don't retry a broken episode forever

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n[worker] interrupted, exiting")
    finally:
        predictor_state.release()
        worker_log.log("worker_shutdown")


if __name__ == "__main__":
    main()
