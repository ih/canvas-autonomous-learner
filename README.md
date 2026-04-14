# canvas-autonomous-learner

Persistent on-robot orchestrator that keeps a canvas world model (CWM) accurate as the scene changes. Idles until prediction error rises, then autonomously collects new data and fine-tunes the model until errors fall again. Repeats indefinitely.

See [PLAN.md](PLAN.md) for the full implementation plan.

## Sibling repos

- `../canvas-world-model` — CWM training/eval (offline)
- `../canvas-robot-control` — MPC control loop + `RobotInterface` + `WorldModelPredictor`
- `../robotic-foundation-model-tests` — LeRobot v3.0 data collection harness

## Python environment

`C:/Projects/pythonenv-lerobot/Scripts/python`

## Run

```
C:/Projects/pythonenv-lerobot/Scripts/python -m learner --config configs/default.yaml
```

Add `--dry-run` to use `DryRunRobotInterface` (no hardware). Add
`--max-iterations N` to bound the outer loop for smoke-testing.

## Tests

```
C:/Projects/pythonenv-lerobot/Scripts/python -m pytest tests/
```

`test_state_machine.py` stubs out verifier/explorer/trainer_driver and drives
the full IDLE -> VERIFY -> EXPLORE -> RETRAIN -> VERIFY -> IDLE trajectory on
a synthetic error trace — no hardware or CWM training stack required.

## Status

Core loop implemented. Hardware soak test pending.
