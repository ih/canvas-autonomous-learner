# Self-Directed CWM Learning Loop — Option A Detailed

## Context

Today the canvas world model (CWM) is learned from **hand-scripted** single-motor datasets: a human runs `run_single_action_record.py` in `robotic-foundation-model-tests`, converts to canvases via `create_dataset.py` in `canvas-world-model`, trains offline with `train_gpt.py` / `train_diffusion.py`, and then `canvas-robot-control` uses the frozen checkpoint for MPC+VLM control. There is no feedback from "did my prediction match reality?" back into data collection or training.

The goal is a persistent background process that:
1. Collects just enough data to predict action outcomes accurately in the current scene.
2. Stops collecting once accuracy is "good enough."
3. Periodically verifies by executing actions and comparing to its own predictions.
4. When a scene change causes prediction errors to rise, resumes collecting + training until the model recovers.
5. Runs indefinitely on the robot without babysitting.

Prediction error on the last-frame visual region (already computed by `../canvas-world-model/evaluate.py`) is the active-learning signal that drives the state machine.

---

## Building blocks already in place

All three pieces of machinery the loop needs exist; they just aren't wired together:

- **Hardware capture + action execution** — in `canvas-robot-control`, `RobotInterface.get_state() -> (cameras, motor_state)` and `RobotInterface.execute_action(int)` already form a complete hardware loop on SO-101 (see `../canvas-robot-control/run_control.py` lines 198-289).
- **World-model inference** — `WorldModelPredictor.predict_batch(context_frame, motor_state, actions, step_size, control_joint_idx, prediction_depth)` in `../canvas-robot-control/control/world_model.py` returns predicted (base, wrist) frames per action. Reuse as-is.
- **Data collection on hardware** — `../robotic-foundation-model-tests/scripts/run_single_action_record.py` records LeRobot v3.0 episodes; append-friendly (Parquet + MP4).
- **Canvas conversion + dataset merge** — `../canvas-world-model/create_dataset.py` + `../canvas-world-model/combine_datasets.py` handle incremental canvas builds and global motor-bound renormalization.
- **Training with resume** — `../canvas-world-model/train_diffusion.py` and `train_gpt.py` support `--resume <ckpt>` fine-tuning (added in commit `f513fef`).
- **Per-action error metric** — `../canvas-world-model/evaluate.py` already computes MSE on the last-frame visual region (`extract_last_frame_visual` around line 64); reuse the same formula online.

The missing piece is an **orchestrator** that owns the state machine and calls into the other repos.

---

## Option A — New repo: `canvas-autonomous-learner` (this repo)

A top-level repo whose only job is the state machine + on-robot scheduler. It depends on the other three repos (as sibling path imports or editable installs), owns no model code, no dataset format code, and no motor drivers of its own.

### Why a separate repo

- `canvas-world-model` is currently pure offline training/eval with no hardware dependency — keep it that way so it stays runnable on any GPU box.
- `canvas-robot-control` is an MPC *controller*; mixing a curator state machine in muddies its purpose and doubles the places that load checkpoints and instantiate robot interfaces.
- `robotic-foundation-model-tests` is a grab-bag of experiments; the autonomous learner is a product, not an experiment.
- A dedicated repo makes it trivial to add a Task Scheduler entry, a config file, and a `runs/` log directory without polluting the others.

### State machine

```
                  +-------------- mean_err > tau_high ----------+
                  v                                             |
   IDLE -- timer --> VERIFY -- mean_err < tau_low --> IDLE      |
    ^                  |                                        |
    |                  +-- mean_err > tau_high --> EXPLORE -----+
    |                                              |
    +------- fine-tune complete & val_mse OK ------+
                            (RETRAIN)
```

- **IDLE** — sleep `idle_seconds`, then transition to VERIFY. Holds the live checkpoint in memory; robot is connected but issuing no motor commands.
- **VERIFY** — pick `K` probes. For each: snapshot (cameras, motor_state), ask CWM to predict next frame for a chosen action, execute that action on hardware, capture the actual next frame, compute MSE on the last-frame visual region. Append (state_hash, action, mse, timestamp) to a rolling window. Update `mean_err` = mean of last `N` probes.
- **EXPLORE** — autonomously collect new episodes using the same recording machinery as `run_single_action_record.py`, but driven by our action-selection policy instead of a uniform scripted reset. Writes to `datasets/autonomous/<session>/` as LeRobot v3.0. Stops after `explore_batch_size` episodes.
- **RETRAIN** — incremental: (1) `create_dataset.py` on the new LeRobot episodes, (2) `combine_datasets.py` to merge with the current canvas dataset, (3) `train_diffusion.py --resume <live_ckpt> --epochs <small> --dataset <merged>`, (4) run `evaluate.py` on a held-out val slice; if val_mse regresses beyond a guard threshold, reject the new checkpoint and fall back. Otherwise atomically swap the live checkpoint via `registry.json`.

### Repo layout

```
canvas-autonomous-learner/
  pyproject.toml              # editable installs of the 3 sibling repos
  README.md
  configs/
    default.yaml              # thresholds, intervals, paths, budgets
  learner/
    __main__.py               # `python -m learner` entry point
    orchestrator.py           # State machine main loop
    states.py                 # IDLE / VERIFY / EXPLORE / RETRAIN enum + transitions
    verifier.py               # Predict -> execute -> compare -> MSE
    explorer.py               # Action-selection policy + episode recording
    trainer_driver.py         # Subprocess wrapper around train_diffusion.py etc
    registry.py               # Live checkpoint + dataset version tracking
    metrics.py                # Rolling error window, per-action histograms
    scene_monitor.py          # Optional: cheap frame-diff change detector
    hardware.py               # Thin adapter over canvas-robot-control RobotInterface
  datasets/                   # (gitignored) autonomously collected LeRobot data
  checkpoints/                # (gitignored) fine-tuned checkpoints + registry.json
  runs/                       # (gitignored) per-session logs, error plots, events
  tests/
    test_state_machine.py     # Transitions under synthetic error traces
    test_verifier_dry.py      # Verifier against DryRunRobotInterface
```

### Key module sketches

**`hardware.py`** — wraps `canvas-robot-control` so the rest of the learner doesn't import from it directly (makes mocking trivial):

```python
from control.robot_interface import RobotInterface, DryRunRobotInterface, JOINTS
from control.world_model import WorldModelPredictor
from control.canvas_utils import stack_cameras_vertically, FRAME_SIZE

class Hardware:
    def __init__(self, cfg, dry_run: bool):
        self.robot = DryRunRobotInterface(...) if dry_run else RobotInterface(...)
        self.predictor = WorldModelPredictor(checkpoint_path=cfg.live_checkpoint, ...)
        self.control_idx = JOINTS.index(cfg.control_joint)

    def observe(self):
        cameras, motor_state = self.robot.get_state()
        return cameras, motor_state, stack_cameras_vertically(
            cameras["base"], cameras["wrist"], FRAME_SIZE)

    def predict(self, context_frame, motor_state, action):
        return self.predictor.predict_batch(
            context_frame, motor_state, [action],
            step_size=self.cfg.step_size, control_joint_idx=self.control_idx,
            prediction_depth=1)[0]

    def execute(self, action):
        self.robot.execute_action(action)

    def reload_checkpoint(self, path):
        self.predictor = WorldModelPredictor(checkpoint_path=path, ...)
        self.predictor.load()
```

**`verifier.py`** — the core error signal:

```python
def verify_once(hw, action, settle_time):
    cams_before, motor_before, ctx = hw.observe()
    pred_base, pred_wrist = hw.predict(ctx, motor_before, action)
    hw.execute(action)
    time.sleep(settle_time)
    cams_after, _, _ = hw.observe()
    mse_base = ((pred_base.astype(np.float32) - cams_after["base"].astype(np.float32)) ** 2).mean()
    mse_wrist = ((pred_wrist.astype(np.float32) - cams_after["wrist"].astype(np.float32)) ** 2).mean()
    return ProbeResult(
        state_key=quantize_motor(motor_before),  # coarse bin for histogramming
        action=action,
        mse=(mse_base + mse_wrist) / 2,
        timestamp=time.time(),
    )
```

This reuses exactly the metric `evaluate.py` already computes offline (last-frame visual MSE), so the online threshold and the offline val MSE speak the same language.

**`explorer.py`** — action selection + episode recording. v0 policy: pick action weighted by recent per-action MSE (highest-error actions get explored first), biased toward motor-position bins with the least recent coverage. Recording reuses `run_single_action_record.py` — preferably refactor its core into an importable function `record_episode(robot, action, duration) -> episode_path` that both the CLI and `explorer.py` call.

**`trainer_driver.py`** — subprocess wrapper; no Python import of train scripts (keeps the learner process light and lets training fail without taking down the loop):

```python
def retrain(cfg, new_lerobot_dir, live_ckpt) -> Optional[Path]:
    # 1. Build canvases from new episodes
    run([cfg.python, "create_dataset.py", "--input", new_lerobot_dir,
         "--output", cfg.canvas_out / "new_batch"])
    # 2. Merge with base canvas dataset
    merged = cfg.canvas_out / f"merged_{stamp()}"
    run([cfg.python, "combine_datasets.py", "--inputs", cfg.base_canvas,
         cfg.canvas_out / "new_batch", "--output", merged])
    # 3. Fine-tune from live checkpoint
    new_ckpt = cfg.ckpt_dir / f"ft_{stamp()}.pth"
    run([cfg.python, "train_diffusion.py", "--dataset", merged,
         "--resume", live_ckpt, "--epochs", str(cfg.ft_epochs),
         "--output", new_ckpt])
    # 4. Evaluate guard
    val_mse = run_and_parse_eval(new_ckpt, cfg.val_dataset)
    if val_mse > cfg.val_guard * baseline_val_mse():
        return None  # regression — keep old checkpoint
    return new_ckpt
```

**`registry.py`** — single source of truth for which checkpoint and which canvas dataset are "live":

```json
{
  "live_checkpoint": "checkpoints/ft_20260412_141203.pth",
  "base_canvas_dataset": "datasets/canvas/base_v1",
  "merged_canvas_dataset": "datasets/canvas/merged_20260412_141203",
  "baseline_val_mse": 0.00412,
  "last_retrain": "2026-04-12T14:12:03",
  "history": [ ... ]
}
```

Updated atomically (write-to-temp + rename) so a crash mid-retrain can't leave a half-swapped pointer.

**`orchestrator.py`** — the main loop is deliberately boring:

```python
def main_loop(cfg):
    hw = Hardware(cfg, dry_run=cfg.dry_run)
    hw.robot.connect()
    hw.predictor.load()
    state = State.IDLE
    window = RollingWindow(cfg.window_size)
    try:
        while not shutdown_requested():
            if state == State.IDLE:
                time.sleep(cfg.idle_seconds)
                state = State.VERIFY
            elif state == State.VERIFY:
                for _ in range(cfg.probes_per_verify):
                    window.add(verifier.verify_once(hw, pick_probe_action(window), cfg.settle_time))
                mean_err = window.mean()
                log_event("verify", mean_err=mean_err)
                if mean_err > cfg.tau_high:
                    state = State.EXPLORE
                else:
                    state = State.IDLE
            elif state == State.EXPLORE:
                new_dir = explorer.collect_batch(hw, cfg.explore_batch_size, window)
                state = State.RETRAIN
            elif state == State.RETRAIN:
                new_ckpt = trainer_driver.retrain(cfg, new_dir, registry.live_checkpoint())
                if new_ckpt:
                    registry.swap(new_ckpt)
                    hw.reload_checkpoint(new_ckpt)
                window.clear()
                state = State.VERIFY
    finally:
        hw.robot.disconnect()
```

### Configuration (`configs/default.yaml`)

```yaml
paths:
  canvas_world_model: ../canvas-world-model
  canvas_robot_control: ../canvas-robot-control
  robotic_foundation_model_tests: ../robotic-foundation-model-tests
  base_canvas: ../canvas-world-model/local/datasets/shoulder_pan_500
  val_dataset: ../canvas-world-model/local/datasets/shoulder_pan_val
  ckpt_dir: ./checkpoints
  canvas_out: ./datasets/canvas
  lerobot_out: ./datasets/lerobot

robot:
  port: COM3
  robot_id: my_so101_follower
  control_joint: shoulder_pan
  step_size: 10.0
  joint_min: -60
  joint_max: 60
  base_camera: 1
  wrist_camera: 0

thresholds:
  tau_low: 0.005       # MSE — tune against evaluate.py offline numbers
  tau_high: 0.015
  val_guard: 1.25      # reject retrain if val_mse > 1.25 * baseline

cadence:
  idle_seconds: 60
  probes_per_verify: 6
  window_size: 24
  settle_time: 0.5
  explore_batch_size: 30   # episodes per EXPLORE burst
  ft_epochs: 3
```

### External dependencies on sibling repos

| Repo | Used for | Coupling |
|---|---|---|
| `canvas-robot-control` | `RobotInterface`, `WorldModelPredictor`, canvas utils | Python import |
| `canvas-world-model` | `create_dataset.py`, `combine_datasets.py`, `train_diffusion.py`, `evaluate.py` | Subprocess CLI |
| `robotic-foundation-model-tests` | Episode recording core (needs small refactor to expose `record_episode()`) | Python import after refactor |

The only upstream change required is **factoring out `record_episode()`** from `run_single_action_record.py` so the explorer can call it without shelling out. Everything else works with existing APIs.

### Python environment

Use the existing virtualenv for all Python commands:
`C:/Projects/pythonenv-lerobot/Scripts/python`

---

## Alternatives considered (not chosen)

- **Option B — add a `learner/` module inside `canvas-robot-control`.** Reuses `RobotInterface` and `WorldModelPredictor` without cross-repo gymnastics, but conflates MPC control with autonomous curation and still has to shell out to `canvas-world-model` for training.
- **Option C — single script `autonomous_learner.py` at the root of `canvas-world-model`.** Fastest v0 but forces `canvas-world-model` to grow a hardware dependency it currently doesn't have.

Option A is worth the extra repo because the state machine will accrue features (scene-change detection, error-weighted exploration, logging dashboards, checkpoint registry, multi-scene support) that don't belong in any of the existing repos.

---

## Open design decisions to lock before building

1. **Exploration policy** — uniform-random actions, or error-weighted (sample (state-bin, action) with highest recent MSE)? Start uniform-random; add weighting once we have baseline numbers.
2. **Scene-change detection** — rely purely on rising verification error (simple, reactive), or add a cheap camera-diff trigger that forces a VERIFY burst when pixels change (faster reaction)? Skip in v0, add if reaction time is a problem.
3. **Training cadence** — fine-tune after every EXPLORE burst, or only when verification error crosses `tau_high` again post-collection? Current plan: one EXPLORE -> one RETRAIN -> re-VERIFY.
4. **Catastrophic-forgetting guard** — the `val_guard` check on the held-out base val split is the minimum; do we also want a replay buffer that always mixes old data into fine-tunes? Rely on `combine_datasets.py` already including the base dataset in every merge.
5. **Diffusion sample variance as a second signal?** — multiple forward passes of the diffusion model give aleatoric uncertainty for free; could drive exploration even before errors rise. Note for v1.

---

## Verification plan

- **State-machine unit tests** — feed synthetic error traces into `orchestrator` and assert it transitions IDLE -> VERIFY -> EXPLORE -> RETRAIN -> VERIFY -> IDLE as errors rise and fall.
- **Dry-run end-to-end** — use `DryRunRobotInterface` + a fake "scene change" (swap the checkpoint the verifier loads) to confirm the whole loop transitions and retrain-subprocesses run without hardware.
- **Single-object hardware soak** — start from a trained checkpoint, let the learner idle, move one object, confirm verification MSE spikes, confirm EXPLORE collects new episodes, confirm RETRAIN lowers MSE back below `tau_low` within a budget of episodes. Visually inspect one probe's (predicted, actual) pair per VERIFY pass.
- **Overnight soak** — leave it running, change the scene a few times, inspect `runs/` for a healthy error-over-time trace (spikes followed by recovery) and no checkpoint regressions logged in `registry.history`.
- **Regression guard** — after each RETRAIN, `evaluate.py` on the base val split must stay within `val_guard` of baseline or the new checkpoint is rejected.

---

## Critical files

**New (this repo):** everything under `canvas-autonomous-learner/learner/` above.

**Upstream change required:**
- `../robotic-foundation-model-tests/scripts/run_single_action_record.py` — extract a `record_episode(robot, action, duration, output_dir) -> Path` function from the CLI body so `explorer.py` can import it. Keep the existing CLI as a thin wrapper.

**Read-only dependencies (no changes):**
- `../canvas-robot-control/control/robot_interface.py` — `RobotInterface`, `DryRunRobotInterface`, `JOINTS`
- `../canvas-robot-control/control/world_model.py` — `WorldModelPredictor.predict_batch`
- `../canvas-robot-control/control/canvas_utils.py` — `stack_cameras_vertically`, `FRAME_SIZE`
- `../canvas-world-model/create_dataset.py` — subprocess CLI
- `../canvas-world-model/combine_datasets.py` — subprocess CLI
- `../canvas-world-model/train_diffusion.py` — subprocess CLI, `--resume` supported
- `../canvas-world-model/evaluate.py` — subprocess CLI, source of the MSE formula reused online
