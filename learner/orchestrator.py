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

from . import claude_advisor, explorer, novelty, trainer_driver, verifier
from .budget import dynamic_explore_batch_size
from .events import EventLog
from .hardware import Hardware
from .metrics import RollingWindow
from .plateau import plateau_reached
from .range_tracker import CurriculumState
from .registry import Registry
from .runtime_knobs import RuntimeKnobs
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
    _run_advisor=None,
) -> dict:
    """Run the unified autonomous learner loop.

    Returns a dict summarizing the termination reason + final state.
    Test-only kwargs `_collect_batch` / `_build_canvases` / `_retrain` /
    `_run_advisor` let `tests/test_state_machine.py` inject fakes
    without touching hardware, subprocesses, or Claude.
    """
    shutdown = _Shutdown()
    _install_signal_handlers(shutdown)
    stop = shutdown_check or (lambda: shutdown.requested)

    # Ensure every per-run dir exists so a fresh config whose run root
    # does not yet exist (e.g. a new experiment arm) just works without
    # requiring the user to pre-create directories.
    for _attr in ("runs_dir", "ckpt_dir", "canvas_out"):
        _p = getattr(cfg.paths, _attr, None)
        if _p:
            Path(_p).mkdir(parents=True, exist_ok=True)

    if registry is None:
        registry = Registry(cfg.paths.registry_file)
    # Pin session name across restarts so dashboard history (rolling MSE
    # chart, wall-clock duration, event stream) survives a kill/resume.
    session_name = registry.session_name()
    if session_name is None:
        session_name = time.strftime("%Y%m%d_%H%M%S")
        registry.set_session_name(session_name)
    if event_log is None:
        event_log = EventLog(cfg.paths.runs_dir, session=session_name)
    examples_dir = Path(cfg.paths.runs_dir) / f"examples_{event_log.session}"
    events_path = Path(cfg.paths.runs_dir) / f"events_{event_log.session}.jsonl"

    if hardware is None:
        hardware = Hardware(cfg, dry_run=getattr(cfg, "dry_run", False))

    # All recording (EXPLORE + VERIFY) goes through the streaming
    # recorder. The legacy episodic path was retired once the streaming
    # sequencer gained probe_script support — verify probes now use the
    # same camera-capture pipeline as EXPLORE bursts (no DSHOW buffer
    # crosstalk, single canvas-format code path). Test-only
    # `_collect_batch` injection bypasses to a fake.
    collect_batch = _collect_batch or explorer.collect_batch_continuous
    build_canvases = _build_canvases or trainer_driver.build_canvases
    retrain_fn = _retrain or trainer_driver.retrain_cumulative
    run_advisor = _run_advisor or claude_advisor.run_advisor

    # -------------------------------------------------------- config hoists
    control_joint = cfg.robot.control_joint
    control_joint_idx = _control_joint_idx(cfg)
    candidates = list(cfg.actions.candidates)

    knobs = RuntimeKnobs.from_cfg(cfg)

    claude_advisor_enabled = bool(
        getattr(cfg.cadence, "claude_advisor_enabled", True)
    )
    claude_max_consecutive_retrains = int(
        getattr(cfg.cadence, "claude_max_consecutive_retrains", 5)
    )
    claude_advisor_crash_timeout_s = float(
        getattr(cfg.cadence, "claude_advisor_crash_timeout_s", 1800.0)
    )
    claude_advisor_model = getattr(cfg.cadence, "claude_advisor_model", None)
    claude_advisor_effort = getattr(cfg.cadence, "claude_advisor_effort", None)
    scene_ready_flag = Path(cfg.paths.runs_dir) / "scene_ready.flag"

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
    # Latest per-(joint, position-bin) MSE breakdown from the most recent
    # accepted RETRAIN. Consumed by plan_per_joint_sub_bursts when
    # cadence.per_joint_targeting is true. None means fall back to the 1D
    # planner (cold start, older checkpoints, or feature disabled).
    current_per_cell_mse: dict | None = None

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

    # Honor any advisor decision that was persisted across a prior crash.
    pending = registry.pending_advisor_decision()
    if pending and pending.get("verb") == "terminate":
        event_log.log(
            "experiment_terminate_pending_replay",
            reason=pending.get("reason"),
            persisted_at=pending.get("t"),
            persisted_cycle=pending.get("cycle"),
        )
        registry.set_pending_advisor_decision(None)
        registry.set_experiment_status("terminated")
        return {
            "reason": "claude_terminate_resumed",
            "reason_detail": str(pending.get("reason", "")),
            "cycle": cycle,
            "total_eps": total_eps,
        }

    registry.set_experiment_status("running")
    event_log.log(
        "experiment_start",
        cold_start=prev_ckpt is None,
        starting_cycle=cycle,
        starting_total_eps=total_eps,
        starting_accumulated_dirs=len(accumulated_dirs),
        active_range=list(_active_range(curriculum, cfg)),
        safety_cap=knobs.safety_cap,
        warmup_cycles=knobs.warmup_cycles,
        tau_low=knobs.tau_low,
        tau_high=knobs.tau_high,
        claude_advisor_enabled=claude_advisor_enabled,
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

    window = RollingWindow(knobs.window_size)
    # Everything now enters via THINK. Claude picks the first real state:
    # warm-start → default_next = VERIFY, cold-start → default_next = EXPLORE.
    pending_default_next_state: State = (
        State.VERIFY if prev_ckpt is not None else State.EXPLORE
    )
    state = State.THINK if claude_advisor_enabled else pending_default_next_state
    last_verify_mean_err: Optional[float] = None
    iteration = 0
    new_lerobot_dirs: list[Path] = []
    new_lerobot_episode_counts: list[int] = []
    # Canvas dirs for the most recent EXPLORE, built upfront so the
    # advisor can inspect them at THINK time. RETRAIN reuses these and
    # skips rebuilding. Parallel `_episode_counts` carries the per-dir
    # episode count for the registry's `episodes_collected` accumulator.
    post_explore_canvas_dirs: list[Path] = []
    post_explore_canvas_episode_counts: list[int] = []
    pending_novelty_report: Optional[dict] = None

    # THINK-phase mutable state
    pending_explore_overrides: Optional[dict] = None
    pending_scene_change_description: Optional[str] = None
    pending_scene_change_requested_at: Optional[float] = None
    last_scene_change: Optional[dict] = None
    consecutive_retrains_without_data: int = 0
    claude_force_cold_start: bool = False

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
            if total_eps >= knobs.safety_cap:
                termination["reason"] = "safety_cap_hit"
                event_log.log(
                    "safety_cap_hit", total_eps=total_eps, cap=knobs.safety_cap,
                )
                break
            # Plateau detection — auto-terminate only when the advisor
            # is disabled. When Claude is in the loop it sees the full
            # locked_val_history in every THINK context and decides for
            # itself whether to terminate, tweak knobs, retrain, or
            # explore differently; an automatic break would yank agency
            # away from Claude mid-decision.
            if not claude_advisor_enabled and plateau_reached(
                registry.locked_val_history()
            ):
                termination["reason"] = "plateau_reached"
                event_log.log("plateau_reached", cycle=cycle)
                break

            iteration += 1
            _emit_state_event()

            # ------------------------------------------------------ IDLE
            # Human-in-the-loop wait. Claude has decided the physical scene
            # needs to be rearranged; the learner blocks until the operator
            # hits "Scene ready" on the dashboard (which writes the flag
            # file). No timeout — the user may be away.
            if state == State.IDLE:
                description = pending_scene_change_description or "(no description)"
                event_log.log(
                    "claude_scene_change_requested",
                    cycle=cycle, description=description,
                )
                # Consume any stale flag left over from a prior request.
                try:
                    if scene_ready_flag.exists():
                        scene_ready_flag.unlink()
                except OSError:
                    pass

                acknowledged = _wait_for_scene_ready(scene_ready_flag, stop, event_log)
                if stop():
                    break
                if not acknowledged:
                    # stop() returned True during the wait — unreachable given
                    # the check above, but keep the branch defensive.
                    break

                ack_t = time.time()
                event_log.log(
                    "scene_ready_acknowledged",
                    cycle=cycle,
                    description=description,
                    requested_at=pending_scene_change_requested_at,
                    acknowledged_at=ack_t,
                )
                last_scene_change = {
                    "description": description,
                    "requested_at": pending_scene_change_requested_at,
                    "acknowledged_at": ack_t,
                }
                pending_scene_change_description = None
                pending_scene_change_requested_at = None
                consecutive_retrains_without_data = 0
                # After the human acks, bounce back through THINK so Claude
                # can re-evaluate with the scene updated.
                pending_default_next_state = State.VERIFY
                state = State.THINK
                continue

            # ------------------------------------------------------ THINK
            # Claude-in-the-loop: spawn `claude -p`, parse its decision,
            # apply overrides, route to the next state. Fail-open to the
            # orchestrator's default next state if anything goes wrong.
            if state == State.THINK:
                default_next_state = pending_default_next_state
                default_next_tok = default_next_state.value.lower()

                if not claude_advisor_enabled:
                    state = default_next_state
                    continue

                context = claude_advisor.snapshot_run_context(
                    events_path, registry, cfg, knobs, curriculum,
                    default_next_state=default_next_tok,
                    consecutive_retrains_without_data=consecutive_retrains_without_data,
                    claude_max_consecutive_retrains=claude_max_consecutive_retrains,
                    last_scene_change=last_scene_change,
                    pending_explore_overrides=pending_explore_overrides,
                    pending_novelty_report=pending_novelty_report,
                )
                prompt = claude_advisor.build_think_prompt(context)
                try:
                    advice = run_advisor(
                        prompt,
                        timeout_s=claude_advisor_crash_timeout_s,
                        model=claude_advisor_model,
                        effort=claude_advisor_effort,
                        default_next_state=default_next_tok,
                        add_dir=str(Path(cfg.paths.runs_dir).resolve()),
                        event_log=event_log,
                    )
                except Exception as e:
                    event_log.log("claude_advisor_failed", error=str(e))
                    advice = claude_advisor._fail_open(default_next_tok)

                event_log.log("claude_think", cycle=cycle, advice=advice)

                # 1. Runtime knob overrides.
                applied = knobs.apply_overrides(
                    advice.get("runtime_overrides") or {}, event_log=event_log,
                )
                if applied:
                    event_log.log("claude_runtime_overrides_applied", applied=applied)

                # 2. Curriculum overrides.
                c_ov = advice.get("curriculum_overrides") or {}
                if c_ov and curriculum is not None:
                    applied = claude_advisor.apply_curriculum_overrides(
                        curriculum, c_ov, event_log=event_log,
                    )
                    if applied:
                        registry.save_range_state(curriculum.to_registry_snapshot())
                        event_log.log(
                            "claude_curriculum_overrides_applied", applied=applied,
                        )

                # 3. Training hyperparameter overrides.
                applied = claude_advisor.apply_cfg_overrides(
                    cfg, advice.get("training_overrides") or {},
                    event_log=event_log,
                )
                if applied:
                    event_log.log("claude_training_overrides_applied", applied=applied)

                # 4. Explore overrides — stashed for the EXPLORE branch.
                ex_ov = advice.get("explore_overrides") or {}
                if ex_ov:
                    pending_explore_overrides = dict(ex_ov)
                    event_log.log(
                        "claude_explore_overrides_pending", overrides=ex_ov,
                    )

                # 5. Cold-start flag.
                if advice.get("from_scratch"):
                    claude_force_cold_start = True

                # 6. Route — apply cap + scene-description check.
                scene_desc = advice.get("scene_change_description")
                requested = advice.get("next_state", default_next_tok)
                canonical = claude_advisor.resolve_next_state(
                    requested, default_next_tok,
                    consecutive_retrains_without_data,
                    claude_max_consecutive_retrains,
                    has_scene_description=bool(scene_desc),
                    event_log=event_log,
                )

                if canonical == "terminate":
                    # Persist the terminate decision BEFORE acting on it so
                    # a crash in the tiny window between advisor return and
                    # orchestrator shutdown can be replayed on resume.
                    registry.set_pending_advisor_decision({
                        "verb": "terminate",
                        "reason": str(advice.get("reason", "")),
                        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "cycle": cycle,
                    })
                    termination["reason"] = "claude_terminate"
                    termination["reason_detail"] = str(advice.get("reason", ""))
                    break

                if canonical == "retrain":
                    new_lerobot_dirs = []
                    new_lerobot_episode_counts = []
                    consecutive_retrains_without_data += 1
                    state = State.RETRAIN
                elif canonical == "idle":
                    pending_scene_change_description = str(scene_desc or "")
                    pending_scene_change_requested_at = time.time()
                    state = State.IDLE
                elif canonical == "explore":
                    state = State.EXPLORE
                else:  # "verify"
                    state = State.VERIFY
                continue

            # ---------------------------------------------------- VERIFY
            if state == State.VERIFY:
                active = _active_range(curriculum, cfg)
                consecutive_retrains_without_data = 0
                if stop():
                    break

                # Drive all probes through the same recorder pipeline
                # that EXPLORE uses: one subprocess, N episodes, each
                # episode forced into its error-weighted (start_pos,
                # direction) script. Each episode is then replayed
                # through canvas-world-model's load_episode +
                # canvas_builder so verify canvases are byte-identical
                # to training canvases.
                try:
                    probes = verifier.verify_batch(
                        cfg,
                        hardware=hardware,
                        window=window,
                        curriculum=curriculum,
                        prev_ckpt=prev_ckpt,
                        cycle=cycle,
                        examples_dir=examples_dir,
                        event_log=event_log,
                        num_probes=knobs.probes_per_verify,
                    )
                except Exception as e:
                    event_log.log("verify_exception", cycle=cycle, error=str(e))
                    probes = []
                for probe in probes:
                    window.add(probe)
                successful_probes = len(probes)

                # If every probe in the burst failed, we have no fresh
                # signal — skip the state-machine decision entirely to
                # avoid acting on stale window data. Bounce through
                # THINK so Claude can decide what to do.
                if successful_probes == 0:
                    event_log.log("verify_burst_empty", cycle=cycle)
                    pending_default_next_state = State.VERIFY
                    state = State.THINK
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
                    tau_low=knobs.tau_low,
                    tau_high=knobs.tau_high,
                )

                # Compute the *default* next state the way the legacy
                # tau-gated logic would. Claude sees this in the THINK
                # context and can override it.
                default_next = State.VERIFY
                if cycle < knobs.warmup_cycles:
                    default_next = State.EXPLORE
                elif mean_err < knobs.tau_low:
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
                            default_next = State.EXPLORE
                        elif curriculum.should_transition_to_secondary():
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
                            default_next = State.EXPLORE
                        elif curriculum.is_done():
                            termination["reason"] = "satisfied_at_full_curriculum"
                            event_log.log(
                                "satisfied_at_full_curriculum",
                                cycle=cycle, total_eps=total_eps,
                                stage=curriculum.stage,
                            )
                            break
                        else:
                            default_next = State.VERIFY
                elif mean_err > knobs.tau_high:
                    if curriculum is not None:
                        curriculum.bad_cycle()
                        registry.save_range_state(curriculum.to_registry_snapshot())
                    default_next = State.EXPLORE

                pending_default_next_state = default_next
                state = State.THINK if claude_advisor_enabled else default_next
                continue

            # ---------------------------------------------------- EXPLORE
            if state == State.EXPLORE:
                consecutive_retrains_without_data = 0

                # Consume any Claude-supplied per-call overrides.
                explore_ov = pending_explore_overrides or {}
                pending_explore_overrides = None
                override_burst = explore_ov.get("num_episodes")
                override_max_sub_bursts = explore_ov.get("max_sub_bursts")
                override_randomize = explore_ov.get("randomize_primary_start")

                if override_burst is not None:
                    burst = int(override_burst)
                else:
                    burst = dynamic_explore_batch_size(
                        mean_err=last_verify_mean_err,
                        tau_high=knobs.tau_high,
                        base=knobs.base_burst,
                        lo=knobs.burst_min,
                        hi=knobs.burst_max,
                    )
                eff_max_sub_bursts = (
                    int(override_max_sub_bursts)
                    if override_max_sub_bursts is not None
                    else knobs.max_sub_bursts
                )
                active = _active_range(curriculum, cfg)
                active_idx = _active_joint_idx(curriculum, control_joint_idx)
                active_jname = _active_joint_name(curriculum, control_joint)

                # Per-joint targeting: when enabled, branch the planner.
                # Returns `(n_eps, joint_name, range)` triples; the
                # collect_batch loop below handles both shapes.
                per_joint_targeting = bool(getattr(
                    getattr(cfg, "cadence", None), "per_joint_targeting", False
                ))
                joint_pool: list[str] = []
                if per_joint_targeting:
                    raw_pool = getattr(cfg.explore, "joints", None)
                    if raw_pool:
                        joint_pool = [
                            str(j).replace(".pos", "") for j in raw_pool
                        ]
                joint_full_ranges: dict[str, tuple[float, float]] = {}
                if per_joint_targeting:
                    cfg_jr = getattr(cfg.explore, "joint_ranges", None)
                    if cfg_jr is not None:
                        src = vars(cfg_jr) if hasattr(cfg_jr, "__dict__") else dict(cfg_jr)
                        for k, v in src.items():
                            if v is None:
                                continue
                            joint_full_ranges[str(k).replace(".pos", "")] = (
                                float(v[0]), float(v[1]),
                            )

                if (
                    per_joint_targeting
                    and len(joint_pool) > 1
                    and current_per_cell_mse is not None
                ):
                    per_joint_bursts = explorer.plan_per_joint_sub_bursts(
                        per_cell_mse=current_per_cell_mse,
                        joint_pool=joint_pool,
                        joint_ranges=joint_full_ranges,
                        total_episodes=burst,
                        max_sub_bursts=eff_max_sub_bursts,
                        min_sub_burst_size=knobs.min_sub_burst_size,
                        min_sub_burst_width=2.0 * float(cfg.robot.step_size),
                    )
                    # Adapt to the (n_eps, range) shape for the loop below
                    # while remembering per-burst joint pinning.
                    sub_bursts = [(n, (r[0], r[1])) for n, _, r in per_joint_bursts]
                    sub_burst_force_joints: list[str | None] = [
                        j for _, j, _ in per_joint_bursts
                    ]
                    event_log.log(
                        "explore_sub_bursts_planned",
                        cycle=cycle,
                        burst=burst,
                        stage=curriculum.stage if curriculum is not None else None,
                        active_joint=active_jname,
                        mode="per_joint",
                        sub_bursts=[
                            {"n_eps": int(n), "joint": j,
                             "range": [float(r[0]), float(r[1])]}
                            for n, j, r in per_joint_bursts
                        ],
                        active_range=list(active),
                    )
                else:
                    sub_bursts = explorer.plan_explore_sub_bursts(
                        window,
                        active,
                        control_joint_idx=active_idx,
                        total_episodes=burst,
                        max_sub_bursts=eff_max_sub_bursts,
                        min_sub_burst_size=knobs.min_sub_burst_size,
                        # Each sub-burst must be wide enough for single_action
                        # to execute a full ±position_delta step without being
                        # flagged as a no-op by its bound-forcing logic.
                        min_sub_burst_width=2.0 * float(cfg.robot.step_size),
                    )
                    sub_burst_force_joints = [None] * len(sub_bursts)
                    event_log.log(
                        "explore_sub_bursts_planned",
                        cycle=cycle,
                        burst=burst,
                        stage=curriculum.stage if curriculum is not None else None,
                        active_joint=active_jname,
                        mode="per_active_joint" if per_joint_targeting else "default",
                        sub_bursts=[
                            {"n_eps": int(n), "range": [float(r[0]), float(r[1])]}
                            for n, r in sub_bursts
                        ],
                        active_range=list(active),
                    )

                new_lerobot_dirs = []
                new_lerobot_episode_counts = []

                # Park the locked inactive joints to Arm A's centers so
                # recorded data matches Arm A's distribution for the joints
                # that single_action doesn't actively sweep. This is a
                # one-shot goto per EXPLORE state, not per sub-burst —
                # the recording subprocess will hold them constant from
                # there on via `lock_inactive_joints: true`.
                inactive_positions = getattr(cfg.explore, "inactive_joint_positions", None)
                if inactive_positions is not None:
                    inactive_dict = (
                        {k: float(v) for k, v in vars(inactive_positions).items()}
                        if hasattr(inactive_positions, "__dict__")
                        else {k: float(v) for k, v in dict(inactive_positions).items()}
                    )
                    try:
                        hardware.goto_home(inactive_dict)
                        event_log.log(
                            "inactive_joints_parked", positions=inactive_dict,
                        )
                    except Exception as e:
                        event_log.log(
                            "inactive_joints_park_failed", error=str(e),
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
                    # Per-joint targeting pins this sub-burst to one
                    # specific joint; the override key follows that
                    # joint, not the curriculum's "active" joint.
                    force_joint = sub_burst_force_joints[sub_idx] if sub_idx < len(sub_burst_force_joints) else None
                    burst_override = dict(base_override)
                    if force_joint is not None:
                        burst_override[f"{force_joint}.pos"] = (
                            float(sub_range[0]), float(sub_range[1]),
                        )
                    else:
                        burst_override[active_joint_key] = (
                            float(sub_range[0]), float(sub_range[1]),
                        )
                    for attempt in range(knobs.explore_max_retries + 1):
                        if home_dict:
                            try:
                                hardware.goto_home(home_dict)
                            except Exception:
                                pass
                        hardware.disconnect()
                        try:
                            collect_kwargs = dict(
                                window=window,
                                event_log=event_log,
                                joint_range_override=burst_override,
                                randomize_primary_start=(
                                    True if override_randomize is None
                                    else bool(override_randomize)
                                ),
                            )
                            if force_joint is not None:
                                collect_kwargs["force_joint"] = force_joint
                            dataset_dir = collect_batch(
                                cfg,
                                int(n_eps),
                                **collect_kwargs,
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
                        if attempt < knobs.explore_max_retries:
                            event_log.log(
                                "explore_retry",
                                cycle=cycle, sub_idx=sub_idx,
                                attempt=attempt + 1,
                                max_attempts=knobs.explore_max_retries + 1,
                                backoff_s=knobs.explore_retry_backoff,
                            )
                            time.sleep(knobs.explore_retry_backoff)
                    if dataset_dir is not None:
                        new_lerobot_dirs.append(Path(dataset_dir))
                        # Track episode count per lerobot dir so the
                        # registry can persist a cumulative episode total
                        # across launches (otherwise total_eps resets to
                        # 0 every process restart and the dashboard's
                        # locked-val trajectory bunches up at the same x
                        # positions). Parallel list — index aligns with
                        # new_lerobot_dirs.
                        new_lerobot_episode_counts.append(int(n_eps))
                        total_eps += int(n_eps)

                if not new_lerobot_dirs:
                    event_log.log("explore_failed_all_retries", cycle=cycle)
                    termination["reason"] = "explore_failed"
                    break

                # Build canvases NOW (before THINK) so the advisor can
                # inspect them and a novelty report can be computed.
                # The RETRAIN branch will reuse these (skip rebuild).
                canvas_out = Path(cfg.paths.canvas_out)
                try:
                    for d, n_eps_for_d in zip(new_lerobot_dirs, new_lerobot_episode_counts):
                        out_dir = canvas_out / f"batch_{_stamp()}_c{cycle}_{d.name}"
                        build_canvases(cfg, d, out_dir, event_log=event_log)
                        post_explore_canvas_dirs.append(out_dir)
                        post_explore_canvas_episode_counts.append(n_eps_for_d)
                except Exception as e:
                    event_log.log("build_canvases_failed", error=str(e), cycle=cycle)
                    termination["reason"] = "build_canvases_failed"
                    break

                # Novelty report: compare new canvas dirs to prior
                # accumulated dirs BEFORE appending the new ones.
                try:
                    pending_novelty_report = novelty.compute_novelty_report(
                        new_canvas_dirs=post_explore_canvas_dirs,
                        prior_canvas_dirs=[Path(p) for p in accumulated_dirs],
                    )
                    event_log.log(
                        "novelty_report",
                        mean_frame_mse_vs_prior_latest=(
                            pending_novelty_report.get("mean_frame_mse_vs_prior_latest")
                        ),
                        num_new_dirs=pending_novelty_report.get("num_new_dirs"),
                        num_prior_dirs=pending_novelty_report.get("num_prior_dirs"),
                    )
                except Exception as e:
                    event_log.log("novelty_report_failed", error=str(e))
                    pending_novelty_report = None

                # Now register the new canvas dirs so subsequent
                # retrains see them. Pass per-dir episode counts so
                # registry.episodes_collected accumulates across launches
                # (otherwise total_eps resets to 0 on every restart and
                # the dashboard's locked-val trajectory clusters at the
                # same x positions for every cycle).
                for d, n_eps_for_d in zip(
                    post_explore_canvas_dirs, post_explore_canvas_episode_counts,
                ):
                    accumulated_dirs.append(str(d))
                    registry.append_canvas_dir(d, episodes_added=n_eps_for_d)
                registry.save_range_state(
                    curriculum.to_registry_snapshot() if curriculum else {}
                )

                # Route through THINK so the advisor can decide whether
                # to retrain immediately (accept the new data), explore
                # more (redundant batch, collect different coverage),
                # or take a different path entirely.
                pending_default_next_state = State.RETRAIN
                state = State.THINK
                continue

            # ---------------------------------------------------- RETRAIN
            if state == State.RETRAIN:
                # If the most recent EXPLORE pre-built canvases (new
                # flow), skip rebuild and jump straight to training.
                # Otherwise (advisor-forced retrain with no new data,
                # or legacy path) build canvases here as before.
                if post_explore_canvas_dirs:
                    # Already built and registered at end of EXPLORE.
                    # Clear so we don't re-use them on subsequent
                    # retrains.
                    post_explore_canvas_dirs = []
                    post_explore_canvas_episode_counts = []
                else:
                    new_canvas_dirs: list[Path] = []
                    new_canvas_episode_counts: list[int] = []
                    canvas_out = Path(cfg.paths.canvas_out)
                    try:
                        for d, n_eps_for_d in zip(
                            new_lerobot_dirs, new_lerobot_episode_counts,
                        ):
                            out_dir = canvas_out / f"batch_{_stamp()}_c{cycle}_{d.name}"
                            build_canvases(cfg, d, out_dir, event_log=event_log)
                            new_canvas_dirs.append(out_dir)
                            new_canvas_episode_counts.append(n_eps_for_d)
                    except Exception as e:
                        event_log.log("build_canvases_failed", error=str(e), cycle=cycle)
                        termination["reason"] = "build_canvases_failed"
                        break

                    for d, n_eps_for_d in zip(new_canvas_dirs, new_canvas_episode_counts):
                        accumulated_dirs.append(str(d))
                        registry.append_canvas_dir(d, episodes_added=n_eps_for_d)
                    registry.save_range_state(
                        curriculum.to_registry_snapshot() if curriculum else {}
                    )

                epochs, from_scratch = _cycle_epochs(cfg, cycle)
                # Claude-forced cold start overrides cycle 0's auto-cold.
                if claude_force_cold_start:
                    from_scratch = True
                    claude_force_cold_start = False
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

                # Recoverable training aborts: VRAM pressure, stall, or
                # hard timeout. Don't terminate — route back to THINK so
                # Claude can shrink batch_size / architecture and retry.
                if isinstance(result, dict) and (
                    result.get("memory_abort")
                    or result.get("stalled")
                    or result.get("timeout")
                ):
                    if result.get("memory_abort"):
                        kind = "memory_abort"
                    elif result.get("stalled"):
                        kind = "stalled"
                    else:
                        kind = "timeout"
                    event_log.log(
                        f"retrain_{kind}_routed_to_think",
                        cycle=cycle,
                        tag=result.get("tag"),
                        summary=result.get("summary"),
                        seconds_since_last_progress=result.get(
                            "seconds_since_last_progress"
                        ),
                        timeout_s=result.get("timeout_s"),
                    )
                    registry.append_locked_val(
                        cycle=cycle, total_eps=total_eps,
                        locked_val_mse=None, train_val_mse=None, accepted=False,
                    )
                    new_lerobot_dirs = []
                    new_lerobot_episode_counts = []
                    pending_default_next_state = State.VERIFY
                    state = (
                        State.THINK if claude_advisor_enabled else State.VERIFY
                    )
                    continue

                train_val_mse = result.get("train_val_mse")
                locked_val_mse = result.get("locked_val_mse")
                new_ckpt = result["checkpoint"]

                # Guard: use locked val preferentially; fall back to train val.
                accepted = True
                guard_signal = "locked"
                if cycle < knobs.warmup_cycles:
                    accepted = True
                elif prev_locked_val_mse is not None and locked_val_mse is not None:
                    if locked_val_mse > knobs.val_guard * prev_locked_val_mse:
                        accepted = False
                elif prev_train_val_mse is not None and train_val_mse is not None:
                    guard_signal = "train"
                    if train_val_mse > knobs.val_guard * prev_train_val_mse:
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
                    # Capture the per-cell MSE breakdown for the next
                    # EXPLORE phase's per-joint sub-burst targeting.
                    # None for older eval runs without per_cell_mse output.
                    current_per_cell_mse = result.get("per_cell_mse")
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
                        guard=knobs.val_guard, consecutive=rejections,
                    )
                    if rejections >= knobs.max_consecutive_rejections:
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
                new_lerobot_episode_counts = []
                # After every retrain, hand the result to Claude so the
                # THINK phase can decide whether to retrain again (with
                # different hparams), verify, explore, or terminate.
                pending_default_next_state = State.VERIFY
                state = (
                    State.THINK if claude_advisor_enabled else State.VERIFY
                )
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
        # Clear any pending advisor decision once we've cleanly shut down —
        # a subsequent startup should not replay a terminate that has
        # already been honored.
        if registry.pending_advisor_decision() is not None:
            registry.set_pending_advisor_decision(None)
        event_log.log("experiment_done", **termination)
        event_log.log("shutdown")

    return termination


def _wait_for_scene_ready(
    flag: Path,
    stop: Callable[[], bool],
    event_log,
    heartbeat_s: float = 30.0,
) -> bool:
    """Block until `flag` appears (dashboard 'Scene ready' button) or
    `stop()` returns True. Polls in 1s slices so SIGINT stays responsive.
    Emits `idle_waiting_for_scene_ready` heartbeats every `heartbeat_s`
    seconds for liveness on the dashboard event stream.

    Returns True on acknowledgment, False if stop() fired first. The
    flag file is deleted on acknowledgment so it can't re-fire a future
    scene request.
    """
    start = time.time()
    last_heartbeat = start
    while not stop():
        if flag.exists():
            try:
                flag.unlink()
            except OSError:
                pass
            return True
        now = time.time()
        if now - last_heartbeat >= heartbeat_s:
            event_log.log(
                "idle_waiting_for_scene_ready",
                elapsed_s=round(now - start, 1),
            )
            last_heartbeat = now
        time.sleep(1.0)
    return False
