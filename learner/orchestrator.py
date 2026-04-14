"""Unified autonomous-learner state machine.

Handles both cold-start (random-init, no prior checkpoint) and
adapt-from-base (pre-trained checkpoint + new data) via the same loop.
VERIFY is always in the decision path: EXPLORE triggers only when the
rolling probe error exceeds `tau_high`, and the loop reaches a clean
"satisfied at full range" termination when the curriculum has stabilized
at the full joint range.

Three curation mechanisms on top of the base state machine:
1. Dynamic `explore_batch_size` scaled by `mean_err / tau_high`.
2. Progressive state-space expansion via `RangeTracker` — start narrow,
   widen only after repeated good verifies in the current range.
3. Error-driven VERIFY probe starting state (`pick_probe_state`) AND
   error-driven EXPLORE sub-bursting (`plan_explore_sub_bursts`) —
   both bias toward the state-space bins where the model is currently
   worst.

Logs a detailed event stream to `runs/events_<session>.jsonl` that the
live dashboard and post-hoc comparison report both consume.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Callable, Optional

from . import explorer, trainer_driver, verifier
from .budget import dynamic_explore_batch_size
from .events import EventLog
from .hardware import Hardware
from .metrics import RollingWindow
from .plateau import plateau_reached
from .range_tracker import CurriculumState
from .registry import Registry
from .states import State


class _Shutdown:
    def __init__(self):
        self.requested = False

    def request(self, *_):
        self.requested = True


def _install_signal_handlers(shutdown: _Shutdown) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, shutdown.request)
        except (ValueError, OSError):
            pass  # not main thread — skip


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _cycle_epochs(cfg, cycle: int) -> tuple[int, bool]:
    """Return `(epochs, from_scratch)` for the given cycle."""
    if cycle == 0:
        return int(getattr(cfg.cadence, "cold_start_epochs", 300)), True
    return int(getattr(cfg.cadence, "ft_epochs", 5)), False


def _build_curriculum(cfg, registry: Registry) -> Optional[CurriculumState]:
    """Return a CurriculumState if `cfg.range.enabled`, else None.

    Handles BOTH the legacy flat `range:` block (single-joint curriculum)
    and the new nested `range.primary/secondary` form (sequential
    curriculum that transitions from primary to secondary once the
    primary joint is fully expanded + stable).
    """
    r = getattr(cfg, "range", None)
    if r is None or not getattr(r, "enabled", True):
        return None
    return CurriculumState.from_config_or_registry(
        r, registry_snapshot=registry.range_snapshot(),
    )


def _active_range(curriculum: Optional[CurriculumState], cfg) -> tuple[float, float]:
    if curriculum is not None:
        return curriculum.active_range
    return (float(cfg.robot.joint_min), float(cfg.robot.joint_max))


def _active_joint_idx(curriculum: Optional[CurriculumState], fallback_idx: int) -> int:
    if curriculum is not None:
        return curriculum.active_joint_idx
    return fallback_idx


def _active_joint_name(curriculum: Optional[CurriculumState], fallback_name: str) -> str:
    if curriculum is not None:
        return curriculum.active_joint_name
    return fallback_name


def _control_joint_idx(cfg) -> int:
    """Index of the control joint in the standard JOINTS order.

    Kept as a static lookup here so the orchestrator doesn't need a
    connected Hardware to reason about range state during tests.
    """
    JOINTS = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    return JOINTS.index(cfg.robot.control_joint)


def main_loop(
    cfg,
    hardware: Hardware | None = None,
    registry: Registry | None = None,
    event_log: EventLog | None = None,
    shutdown_check: Callable[[], bool] | None = None,
    max_iterations: int | None = None,
    *,
    _collect_batch=None,
    _build_canvases=None,
    _retrain=None,
) -> dict:
    """Run the unified autonomous learner loop.

    Returns a dict summarizing the termination reason + final state.
    Test-only kwargs `_collect_batch` / `_build_canvases` / `_retrain`
    let `tests/test_state_machine.py` inject fakes without touching
    hardware or subprocesses.
    """
    shutdown = _Shutdown()
    _install_signal_handlers(shutdown)
    stop = shutdown_check or (lambda: shutdown.requested)

    if event_log is None:
        event_log = EventLog(cfg.paths.runs_dir)
    examples_dir = Path(cfg.paths.runs_dir) / f"examples_{event_log.session}"

    if registry is None:
        registry = Registry(cfg.paths.registry_file)
    if hardware is None:
        hardware = Hardware(cfg, dry_run=getattr(cfg, "dry_run", False))

    collect_batch = _collect_batch or explorer.collect_batch
    build_canvases = _build_canvases or trainer_driver.build_canvases
    retrain_fn = _retrain or trainer_driver.retrain_cumulative

    # -------------------------------------------------------- config hoists
    control_joint = cfg.robot.control_joint
    control_joint_idx = _control_joint_idx(cfg)
    candidates = list(cfg.actions.candidates)

    tau_low = float(cfg.thresholds.tau_low)
    tau_high = float(cfg.thresholds.tau_high)
    val_guard = float(cfg.thresholds.val_guard)
    max_consecutive_rejections = int(
        getattr(cfg.thresholds, "max_consecutive_rejections", 3)
    )

    idle_seconds = float(getattr(cfg.cadence, "idle_seconds", 10))
    # Linear backoff: each consecutive IDLE sleeps `idle_seconds * n`,
    # reset to 1 on the next EXPLORE or RETRAIN phase.
    idle_backoff_count = 1
    # Dashboard "verify now" button writes this file to force an
    # immediate VERIFY from any IDLE sleep.
    trigger_verify_file = Path(cfg.paths.runs_dir) / "trigger_verify.flag"
    probes_per_verify = int(getattr(cfg.cadence, "probes_per_verify", 4))
    window_size = int(getattr(cfg.cadence, "window_size", 16))
    settle_time = float(getattr(cfg.cadence, "settle_time", 0.5))

    base_burst = int(getattr(cfg.cadence, "base_explore_batch_size",
                             getattr(cfg.cadence, "explore_batch_size", 50)))
    burst_min = int(getattr(cfg.cadence, "explore_batch_size_min", 10))
    burst_max = int(getattr(cfg.cadence, "explore_batch_size_max", 150))
    max_sub_bursts = int(getattr(cfg.cadence, "max_sub_bursts", 3))
    min_sub_burst_size = int(getattr(cfg.cadence, "min_sub_burst_size", 10))

    safety_cap = int(getattr(cfg.cadence, "safety_cap_episodes", 10**9))
    warmup_cycles = int(getattr(cfg.cadence, "warmup_cycles", 2))
    explore_max_retries = int(getattr(cfg.cadence, "explore_max_retries", 2))
    explore_retry_backoff = float(getattr(cfg.cadence, "explore_retry_backoff", 10.0))

    locked_val_dataset = getattr(cfg.paths, "locked_val_dataset", None)
    base_canvas = getattr(cfg.paths, "base_canvas", None)

    # ------------------------------------------------------- resume state
    # Seed the registry from cfg.paths.live_checkpoint on a fresh run so a
    # warm-start config (non-cold-start) actually starts warm. Without this
    # seed, a brand new Registry file always reports `live_checkpoint = None`
    # and the loop would go cold-start even when the user pointed us at a
    # pre-trained checkpoint.
    if (
        registry.live_checkpoint() is None
        and getattr(cfg.paths, "live_checkpoint", None)
    ):
        registry.set_baseline(
            live_checkpoint=cfg.paths.live_checkpoint,
            base_canvas_dataset=cfg.paths.base_canvas or "",
            baseline_val_mse=None,
        )
        event_log.log(
            "registry_seeded",
            live_checkpoint=cfg.paths.live_checkpoint,
            base_canvas=cfg.paths.base_canvas,
        )

    accumulated_dirs: list[str] = registry.accumulated_canvas_dirs()
    total_eps: int = registry.episodes_collected()
    prev_ckpt: str | None = registry.live_checkpoint()
    prev_train_val_mse: float | None = None
    prev_locked_val_mse: float | None = None

    # Seed prev_*_val from registry history (last accepted entry) so a
    # resumed run compares against the last ACCEPTED value, not None.
    for entry in reversed(registry.locked_val_history()):
        if entry.get("accepted") and entry.get("locked_val_mse") is not None:
            prev_locked_val_mse = float(entry["locked_val_mse"])
            if entry.get("train_val_mse") is not None:
                prev_train_val_mse = float(entry["train_val_mse"])
            break
    cycle = len(registry.locked_val_history())

    # If there's a pre-trained base dataset and no accumulated history
    # yet, seed the accumulator with the base so every retrain includes it.
    if base_canvas is not None and not accumulated_dirs:
        accumulated_dirs = [str(base_canvas)]
        for d in accumulated_dirs:
            registry.append_canvas_dir(d, episodes_added=0)

    curriculum = _build_curriculum(cfg, registry)
    if curriculum is not None:
        event_log.log(
            "curriculum_init",
            stage=curriculum.stage,
            active_joint=curriculum.active_joint_name,
            active_range=list(curriculum.active_range),
            primary_full=[curriculum.primary.full_min, curriculum.primary.full_max],
            has_secondary=curriculum.secondary_config is not None,
        )

    registry.set_experiment_status("running")
    event_log.log(
        "experiment_start",
        cold_start=prev_ckpt is None,
        starting_cycle=cycle,
        starting_total_eps=total_eps,
        starting_accumulated_dirs=len(accumulated_dirs),
        active_range=list(_active_range(curriculum, cfg)),
        safety_cap=safety_cap,
        warmup_cycles=warmup_cycles,
        tau_low=tau_low,
        tau_high=tau_high,
    )

    hardware.connect()
    if prev_ckpt is not None:
        hardware.load_predictor(prev_ckpt)

    # Park at home before doing anything else so the arm is in a known
    # safe pose at the start of every run. Used again before retrains and
    # on termination.
    home_ns = getattr(cfg.robot, "home", None)
    home_dict = {k: float(v) for k, v in vars(home_ns).items()} if home_ns else {}
    if home_dict:
        try:
            hardware.goto_home(home_dict)
            event_log.log("goto_home", phase="startup", home=home_dict)
        except Exception as e:
            event_log.log("goto_home_failed", phase="startup", error=str(e))

    window = RollingWindow(window_size)
    # Cold start: skip the opening IDLE and go straight to EXPLORE.
    state = State.IDLE if prev_ckpt is not None else State.EXPLORE
    last_verify_mean_err: Optional[float] = None
    iteration = 0
    new_lerobot_dirs: list[Path] = []

    termination = {
        "reason": "unknown",
        "cycles": cycle,
        "total_eps": total_eps,
        "final_locked_val_mse": prev_locked_val_mse,
        "final_checkpoint": prev_ckpt,
        "final_range": list(_active_range(curriculum, cfg)),
        "final_stage": curriculum.stage if curriculum is not None else None,
    }

    def _emit_state_event() -> None:
        event_log.log(
            "state",
            state=state.value,
            iteration=iteration,
            cycle=cycle,
            total_eps=total_eps,
            active_range=list(_active_range(curriculum, cfg)),
        )

    try:
        while not stop():
            # Keep termination snapshot fresh so break-paths always report
            # the current totals (safety_cap / plateau / max_iterations).
            termination["total_eps"] = total_eps
            termination["cycles"] = cycle
            if curriculum is not None:
                termination["final_range"] = list(curriculum.active_range)
                termination["final_stage"] = curriculum.stage

            if max_iterations is not None and iteration >= max_iterations:
                termination["reason"] = "max_iterations"
                break
            if total_eps >= safety_cap:
                termination["reason"] = "safety_cap_hit"
                event_log.log("safety_cap_hit", total_eps=total_eps, cap=safety_cap)
                break
            if plateau_reached(registry.locked_val_history()):
                termination["reason"] = "plateau_reached"
                event_log.log("plateau_reached", cycle=cycle)
                break

            iteration += 1
            _emit_state_event()

            # ------------------------------------------------------ IDLE
            if state == State.IDLE:
                sleep_for = idle_seconds * idle_backoff_count
                event_log.log(
                    "idle_sleep",
                    seconds=sleep_for,
                    backoff_count=idle_backoff_count,
                )
                triggered = _sleep_cancelable(sleep_for, stop, trigger_verify_file)
                if stop():
                    break
                if triggered:
                    event_log.log("idle_interrupted_by_trigger")
                # Each consecutive IDLE grows the sleep. Reset on EXPLORE/RETRAIN.
                idle_backoff_count += 1
                state = State.VERIFY
                continue

            # ---------------------------------------------------- VERIFY
            if state == State.VERIFY:
                active = _active_range(curriculum, cfg)
                active_joint_idx_for_verify = _active_joint_idx(curriculum, control_joint_idx)
                active_joint_name_for_verify = _active_joint_name(curriculum, control_joint)
                successful_probes = 0
                import random as _rng
                for probe_idx in range(probes_per_verify):
                    if stop():
                        break

                    # Error-driven state pick on the current curriculum joint.
                    target_pos = explorer.pick_probe_state(
                        window, active, control_joint_idx=active_joint_idx_for_verify,
                    )

                    # Reposition the NON-active curriculum joint to a
                    # uniform-random position inside whatever range EXPLORE
                    # is currently using for it (primary.full_range in
                    # stage 2, secondary.pinned_range in stage 1). This
                    # keeps each probe's motor state inside the same
                    # distribution the model was just trained on.
                    if curriculum is not None:
                        ranges = curriculum.joint_range_override() or {}
                        active_key = f"{active_joint_name_for_verify}.pos"
                        for joint_key, (lo, hi) in ranges.items():
                            if joint_key == active_key:
                                continue
                            joint_name = joint_key.replace(".pos", "")
                            sampled = _rng.uniform(float(lo), float(hi))
                            try:
                                hardware.goto(joint_name, sampled)
                            except Exception as e:
                                event_log.log(
                                    "verify_nonactive_goto_failed",
                                    cycle=cycle, joint=joint_name, error=str(e),
                                )

                    action = explorer.pick_probe_action(window, candidates)
                    tag = f"c{cycle:03d}_p{probe_idx}_{time.strftime('%H%M%S')}"
                    try:
                        probe = verifier.verify_once(
                            hardware, action, settle_time,
                            examples_dir=examples_dir,
                            example_tag=tag,
                            target_joint=active_joint_name_for_verify,
                            target_position=target_pos,
                        )
                    except Exception as e:
                        event_log.log(
                            "verify_probe_failed",
                            cycle=cycle, probe_idx=probe_idx, error=str(e),
                        )
                        continue
                    window.add(probe)
                    successful_probes += 1
                    event_log.log(
                        "probe",
                        cycle=cycle,
                        action=probe.action,
                        mse=probe.mse,
                        state_key=probe.state_key,
                        motor_state=list(probe.motor_state) if probe.motor_state else None,
                        target_position=float(target_pos),
                    )

                # If every probe in the burst failed, we have no fresh
                # signal — skip the state-machine decision entirely to
                # avoid acting on stale window data. Go to IDLE and try
                # again later.
                if successful_probes == 0:
                    event_log.log("verify_burst_empty", cycle=cycle)
                    state = State.IDLE
                    continue

                # Only count probes whose start state landed inside the
                # active range toward the gating mean — otherwise drift
                # into/out of the active region skews the signal.
                n_in_range = window.count_in_range(active, control_joint_idx)
                if n_in_range > 0:
                    mean_err = window.mean_in_range(active, control_joint_idx)
                else:
                    mean_err = window.mean()
                last_verify_mean_err = mean_err
                event_log.log(
                    "verify_summary",
                    cycle=cycle,
                    mean_err=mean_err,
                    n_in_range=n_in_range,
                    n_total=len(window),
                    active_range=list(active),
                    tau_low=tau_low,
                    tau_high=tau_high,
                )

                # Decide next state
                if cycle < warmup_cycles:
                    # Warmup: always explore regardless of verify MSE.
                    state = State.EXPLORE
                elif mean_err < tau_low:
                    if curriculum is not None:
                        curriculum.good_cycle()
                        registry.save_range_state(curriculum.to_registry_snapshot())
                        if curriculum.should_expand():
                            old = tuple(curriculum.active_tracker().active)
                            new_range = curriculum.expand(
                                cycle=cycle, total_eps=total_eps,
                            )
                            registry.save_range_state(curriculum.to_registry_snapshot())
                            event_log.log(
                                "range_expanded",
                                cycle=cycle, total_eps=total_eps,
                                stage=curriculum.stage,
                                joint=curriculum.active_joint_name,
                                old_range=list(old), new_range=list(new_range),
                            )
                            state = State.EXPLORE
                        elif curriculum.should_transition_to_secondary():
                            # Primary is at max + stable; graduate to stage 2.
                            old_stage = curriculum.stage
                            new_tracker = curriculum.transition_to_secondary()
                            registry.save_range_state(curriculum.to_registry_snapshot())
                            event_log.log(
                                "curriculum_stage_transition",
                                cycle=cycle, total_eps=total_eps,
                                old_stage=old_stage,
                                new_stage=curriculum.stage,
                                new_joint=new_tracker.control_joint,
                                new_active_range=list(new_tracker.active),
                            )
                            state = State.EXPLORE
                        elif curriculum.is_done():
                            termination["reason"] = "satisfied_at_full_curriculum"
                            event_log.log(
                                "satisfied_at_full_curriculum",
                                cycle=cycle, total_eps=total_eps,
                                stage=curriculum.stage,
                            )
                            break
                        else:
                            state = State.IDLE
                    else:
                        state = State.IDLE
                elif mean_err > tau_high:
                    if curriculum is not None:
                        curriculum.bad_cycle()
                        registry.save_range_state(curriculum.to_registry_snapshot())
                    state = State.EXPLORE
                else:
                    state = State.IDLE
                continue

            # ---------------------------------------------------- EXPLORE
            if state == State.EXPLORE:
                idle_backoff_count = 1
                burst = dynamic_explore_batch_size(
                    mean_err=last_verify_mean_err,
                    tau_high=tau_high,
                    base=base_burst,
                    lo=burst_min,
                    hi=burst_max,
                )
                active = _active_range(curriculum, cfg)
                active_idx = _active_joint_idx(curriculum, control_joint_idx)
                active_jname = _active_joint_name(curriculum, control_joint)
                sub_bursts = explorer.plan_explore_sub_bursts(
                    window,
                    active,
                    control_joint_idx=active_idx,
                    total_episodes=burst,
                    max_sub_bursts=max_sub_bursts,
                    min_sub_burst_size=min_sub_burst_size,
                    # Each sub-burst must be wide enough for single_action
                    # to execute a full ±position_delta step without being
                    # flagged as a no-op by its bound-forcing logic.
                    min_sub_burst_width=2.0 * float(cfg.robot.step_size),
                )
                event_log.log(
                    "explore_sub_bursts_planned",
                    cycle=cycle,
                    burst=burst,
                    stage=curriculum.stage if curriculum is not None else None,
                    active_joint=active_jname,
                    sub_bursts=[
                        {"n_eps": int(n), "range": [float(r[0]), float(r[1])]}
                        for n, r in sub_bursts
                    ],
                    active_range=list(active),
                )

                new_lerobot_dirs = []

                # Park the locked inactive joints to Arm A's centers so
                # recorded data matches Arm A's distribution for the joints
                # that single_action doesn't actively sweep. This is a
                # one-shot goto per EXPLORE state, not per sub-burst —
                # the recording subprocess will hold them constant from
                # there on via `lock_inactive_joints: true`.
                inactive_positions = getattr(cfg.explore, "inactive_joint_positions", None)
                if inactive_positions is not None:
                    for joint_name, target in vars(inactive_positions).items() if hasattr(inactive_positions, "__dict__") else dict(inactive_positions).items():
                        try:
                            hardware.goto(joint_name, float(target))
                            event_log.log(
                                "inactive_joint_parked",
                                joint=joint_name, target=float(target),
                            )
                        except Exception as e:
                            event_log.log(
                                "inactive_joint_park_failed",
                                joint=joint_name, error=str(e),
                            )

                # Build the base joint_range_override from the curriculum
                # state (stage 1 pins secondary at a tight center; stage 2
                # pins primary at full range). We then replace the currently-
                # active joint with the per-sub-burst narrow bin.
                base_override: dict = {}
                if curriculum is not None:
                    base_override = dict(curriculum.joint_range_override())
                active_joint_key = f"{active_jname}.pos"
                for sub_idx, (n_eps, sub_range) in enumerate(sub_bursts):
                    dataset_dir: Optional[Path] = None
                    burst_override = dict(base_override)
                    burst_override[active_joint_key] = (
                        float(sub_range[0]), float(sub_range[1]),
                    )
                    for attempt in range(explore_max_retries + 1):
                        if home_dict:
                            try:
                                hardware.goto_home(home_dict)
                            except Exception:
                                pass
                        hardware.disconnect()
                        try:
                            dataset_dir = collect_batch(
                                cfg,
                                int(n_eps),
                                window=window,
                                event_log=event_log,
                                joint_range_override=burst_override,
                                randomize_primary_start=True,
                            )
                        finally:
                            try:
                                hardware.connect()
                                if prev_ckpt is not None:
                                    hardware.load_predictor(prev_ckpt)
                            except Exception as e:
                                event_log.log(
                                    "hardware_reconnect_failed",
                                    error=str(e), attempt=attempt,
                                )
                        if dataset_dir is not None:
                            if attempt > 0:
                                event_log.log(
                                    "explore_retry_succeeded",
                                    cycle=cycle, sub_idx=sub_idx, attempt=attempt,
                                )
                            break
                        if attempt < explore_max_retries:
                            event_log.log(
                                "explore_retry",
                                cycle=cycle, sub_idx=sub_idx,
                                attempt=attempt + 1,
                                max_attempts=explore_max_retries + 1,
                                backoff_s=explore_retry_backoff,
                            )
                            time.sleep(explore_retry_backoff)
                    if dataset_dir is not None:
                        new_lerobot_dirs.append(Path(dataset_dir))
                        total_eps += int(n_eps)

                if not new_lerobot_dirs:
                    event_log.log("explore_failed_all_retries", cycle=cycle)
                    termination["reason"] = "explore_failed"
                    break

                state = State.RETRAIN
                continue

            # ---------------------------------------------------- RETRAIN
            if state == State.RETRAIN:
                idle_backoff_count = 1
                new_canvas_dirs: list[Path] = []
                canvas_out = Path(cfg.paths.canvas_out)
                try:
                    for d in new_lerobot_dirs:
                        out_dir = canvas_out / f"batch_{_stamp()}_c{cycle}_{d.name}"
                        build_canvases(cfg, d, out_dir, event_log=event_log)
                        new_canvas_dirs.append(out_dir)
                except Exception as e:
                    event_log.log("build_canvases_failed", error=str(e), cycle=cycle)
                    termination["reason"] = "build_canvases_failed"
                    break

                for d in new_canvas_dirs:
                    accumulated_dirs.append(str(d))
                    registry.append_canvas_dir(d, episodes_added=0)
                registry.save_range_state(
                    curriculum.to_registry_snapshot() if curriculum else {}
                )

                epochs, from_scratch = _cycle_epochs(cfg, cycle)
                # Park the arm before training so it doesn't sit in whatever
                # random pose the last episode left it in — keeps the motors
                # holding a safe pose for the duration of training.
                if home_dict:
                    try:
                        hardware.goto_home(home_dict)
                        event_log.log("goto_home", phase="pre_retrain")
                    except Exception as e:
                        event_log.log("goto_home_failed", phase="pre_retrain", error=str(e))
                event_log.log(
                    "retrain_start",
                    cycle=cycle,
                    epochs=epochs,
                    from_scratch=from_scratch,
                    num_accumulated_dirs=len(accumulated_dirs),
                    total_eps=total_eps,
                )
                result = retrain_fn(
                    cfg,
                    accumulated_canvas_dirs=list(accumulated_dirs),
                    resume_checkpoint=None if from_scratch else prev_ckpt,
                    epochs=epochs,
                    locked_val_dataset=locked_val_dataset,
                    event_log=event_log,
                )
                if result is None:
                    event_log.log("retrain_failed", cycle=cycle)
                    registry.append_locked_val(
                        cycle=cycle, total_eps=total_eps,
                        locked_val_mse=None, train_val_mse=None, accepted=False,
                    )
                    termination["reason"] = "retrain_failed"
                    break

                train_val_mse = result.get("train_val_mse")
                locked_val_mse = result.get("locked_val_mse")
                new_ckpt = result["checkpoint"]

                # Guard: use locked val preferentially; fall back to train val.
                accepted = True
                guard_signal = "locked"
                if cycle < warmup_cycles:
                    accepted = True
                elif prev_locked_val_mse is not None and locked_val_mse is not None:
                    if locked_val_mse > val_guard * prev_locked_val_mse:
                        accepted = False
                elif prev_train_val_mse is not None and train_val_mse is not None:
                    guard_signal = "train"
                    if train_val_mse > val_guard * prev_train_val_mse:
                        accepted = False

                if accepted:
                    registry.swap(
                        new_checkpoint=new_ckpt,
                        merged_canvas_dataset=result.get("merged_dataset"),
                        val_mse=train_val_mse,
                    )
                    prev_ckpt = new_ckpt
                    prev_train_val_mse = train_val_mse
                    prev_locked_val_mse = locked_val_mse
                    registry.reset_guard_rejections()
                    try:
                        hardware.reload_checkpoint(prev_ckpt)
                    except Exception as e:
                        event_log.log("reload_checkpoint_failed", error=str(e))
                    event_log.log(
                        "checkpoint_swapped",
                        cycle=cycle, checkpoint=new_ckpt,
                        train_val_mse=train_val_mse,
                        locked_val_mse=locked_val_mse,
                        guard_signal=guard_signal,
                    )
                else:
                    rejections = registry.bump_guard_rejections()
                    event_log.log(
                        "guard_rejected",
                        cycle=cycle, guard_signal=guard_signal,
                        train_val_mse=train_val_mse,
                        locked_val_mse=locked_val_mse,
                        prev_train_val_mse=prev_train_val_mse,
                        prev_locked_val_mse=prev_locked_val_mse,
                        guard=val_guard, consecutive=rejections,
                    )
                    if rejections >= max_consecutive_rejections:
                        event_log.log(
                            "guard_force_swap",
                            cycle=cycle, consecutive=rejections,
                        )
                        registry.swap(
                            new_checkpoint=new_ckpt,
                            merged_canvas_dataset=result.get("merged_dataset"),
                            val_mse=train_val_mse,
                            notes="force_swap_after_max_rejections",
                        )
                        prev_ckpt = new_ckpt
                        prev_train_val_mse = train_val_mse
                        prev_locked_val_mse = locked_val_mse
                        registry.reset_guard_rejections()
                        try:
                            hardware.reload_checkpoint(prev_ckpt)
                        except Exception as e:
                            event_log.log("reload_checkpoint_failed", error=str(e))
                        accepted = True

                registry.append_locked_val(
                    cycle=cycle, total_eps=total_eps,
                    locked_val_mse=locked_val_mse,
                    train_val_mse=train_val_mse,
                    accepted=accepted,
                )
                event_log.log(
                    "locked_val_measured",
                    cycle=cycle, total_eps=total_eps,
                    locked_val_mse=locked_val_mse,
                    train_val_mse=train_val_mse,
                    accepted=accepted,
                )

                termination["cycles"] = cycle + 1
                termination["total_eps"] = total_eps
                termination["final_locked_val_mse"] = locked_val_mse
                termination["final_checkpoint"] = prev_ckpt
                if curriculum is not None:
                    termination["final_range"] = list(curriculum.active_range)
                    termination["final_stage"] = curriculum.stage

                # Clear the rolling verify window after each retrain so the
                # next verify doesn't mix pre- and post-retrain probes (the
                # whole point of retraining is that the model just changed).
                window.clear()

                cycle += 1
                new_lerobot_dirs = []
                state = State.VERIFY
                continue

    finally:
        # Park at home before exit. Deliberately NOT calling hardware.disconnect()
        # here — on SO-101 disconnect disables torque, which drops every joint
        # and the arm can crash under gravity. Letting the process exit with
        # the bus still open leaves torque engaged at the home pose until the
        # next connection or power cycle.
        try:
            if home_dict:
                hardware.goto_home(home_dict)
                event_log.log("goto_home", phase="shutdown")
        except Exception as e:
            event_log.log("goto_home_failed", phase="shutdown", error=str(e))
        registry.set_experiment_status(termination.get("reason", "unknown"))
        event_log.log("experiment_done", **termination)
        event_log.log("shutdown")

    return termination


def _sleep_cancelable(
    seconds: float,
    stop: Callable[[], bool],
    trigger_file: Optional[Path] = None,
) -> bool:
    """Sleep in short slices so SIGINT doesn't wait out a long idle window.

    Returns True if the sleep was cut short by `trigger_file` appearing
    on disk (dashboard "verify now" button); False otherwise. The file
    is consumed (deleted) before returning so it can't re-fire.
    """
    end = time.time() + seconds
    while time.time() < end:
        if stop():
            return False
        if trigger_file is not None and trigger_file.exists():
            try:
                trigger_file.unlink()
            except OSError:
                pass
            return True
        time.sleep(min(0.5, end - time.time()))
    return False
