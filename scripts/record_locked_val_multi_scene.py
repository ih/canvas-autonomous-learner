"""Record a multi-scene corpus for either locked-val or wide-verify seed.

Two modes:

  Default (locked-val mode): records `--num-scenes` independent scenes
    and merges the per-scene canvases into the final
    `locked_val_shoulder` / `locked_val_elbow` paths configured in the
    YAML (via canvas-world-model/combine_datasets.py). Use this for
    legacy-mode experiments (sim_1b, pan_focus_1b) where locked_val_mse
    is the val_guard signal.

  `--seed-wide-verify`: records the same multi-scene structure, but
    instead of combining into one locked-val output, registers each
    per-scene canvas dir in the registry's wide-verify accumulator
    (`registry.append_wide_verify_dir`). Use this for lifelong-mode
    experiments (pan_only_1b) so the held-out cross-scene corpus has
    a non-empty seed before the very first cycle runs. After a few
    in-flight wide-verify passes during training, the seed becomes a
    minor fraction of the corpus.

Both modes use the shared scene-prompting helper at
`learner/scene_prompts.py`, so the operator instructions match what
the live wide-verify mode shows on the dashboard.

Usage:
    python scripts/record_locked_val_multi_scene.py \\
        --config configs/pan_focus_1b.yaml \\
        --num-scenes 5 \\
        --episodes-per-scene 10

Default sizing rationale (5 scenes x 10 ep/joint = 50 ep/joint):
  - The single-scene baseline (record_locked_val.py) records 30 ep/joint
    against ONE scene. 50 ep/joint here is ~67% more total samples,
    which keeps the aggregate MSE estimate at least as tight while
    spreading coverage across 5 distinct object arrangements.
  - 10 ep/scene against shoulder_pan's 120-degree range is ~12 degrees
    per scene - fine enough that within-scene MSE isn't dominated by
    sparse sampling.
  - 5 scenes is the rough plateau: past that, each additional scene's
    marginal contribution to a generalization estimate drops off, while
    operator time continues to grow linearly.
  - Total operator time: ~12-13 min (5 rearrangements at ~30s each +
    ~10 min recording + canvas-build/combine).

Resume after interruption:
    --start-scene N         start at scene N (1-indexed); earlier scenes
                            are assumed already-recorded in the HF cache
    --skip-shoulder         skip all shoulder recordings (already done)
    --skip-elbow            skip all elbow recordings (already done)
    --shoulder-only         record only shoulder_pan, no elbow_flex (the
                            elbow_flex repo_id_prefix is also skipped at
                            canvas-build time so an empty elbow corpus
                            isn't produced)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner import explorer, scene_prompts  # noqa: E402
from learner.config import load_config  # noqa: E402
from learner.registry import Registry  # noqa: E402


_prompt_for_scene = scene_prompts.prompt_terminal


# --------------------------------------------------------------- recording


def _override_explore_cfg(cfg, joint_name: str):
    """Shallow-copy cfg and force single-joint recording on `joint_name`.

    The single_action policy's __post_init__ rejects identical primary
    and secondary joints; its default secondary is elbow_flex.pos, so
    when we record elbow we must re-point the secondary elsewhere.
    Mirrors `_override_explore_cfg` in record_locked_val.py.
    """
    new_cfg = copy(cfg)
    new_cfg.explore = copy(cfg.explore)
    new_cfg.explore.policy_joint_name = joint_name
    new_cfg.explore.vary_target_joint = False
    new_cfg.explore.joints = None
    if joint_name == "elbow_flex.pos":
        new_cfg.explore.secondary_joint_name = "shoulder_pan.pos"
    else:
        new_cfg.explore.secondary_joint_name = "elbow_flex.pos"
    return new_cfg


def _record_one_joint(
    cfg, *, joint_name: str, joint_min: float, joint_max: float,
    num_episodes: int, repo_id_prefix: str, tag: str,
) -> Path:
    sub_cfg = _override_explore_cfg(cfg, joint_name)
    joint_range_override = {joint_name: (float(joint_min), float(joint_max))}
    print(f"\n[{tag}] recording {num_episodes} episodes on {joint_name} "
          f"(starts uniform in [{joint_min}, {joint_max}])")
    dataset_path = explorer.collect_batch_continuous(
        sub_cfg,
        num_episodes=num_episodes,
        joint_range_override=joint_range_override,
        repo_id_prefix=repo_id_prefix,
        event_tag="locked_val_record_start",
        randomize_primary_start=True,
    )
    if dataset_path is None:
        raise RuntimeError(
            f"[{tag}] recorder subprocess failed — see output above."
        )
    print(f"[{tag}] recorded to: {dataset_path}")
    return Path(dataset_path)


# --------------------------------------------------------- canvas build/merge


def _canvas_build(
    cfg, *, lerobot_path: Path, output_dir: Path, motor_bounds: dict, tag: str,
) -> None:
    """Convert one LeRobot dataset to canvas format via create_dataset.py."""
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    if output_dir.exists():
        print(f"[{tag}] wiping stale output dir: {output_dir}")
        shutil.rmtree(output_dir)
    cmd = [
        python_exe,
        "create_dataset.py",
        "--lerobot-path", str(lerobot_path),
        "--output", str(output_dir),
        "--cameras", cfg.explore.base_camera_name, cfg.explore.wrist_camera_name,
        "--stack-cameras", "vertical",
        "--frame-size", "224", "224",
        "--motor-bounds-json", json.dumps(motor_bounds),
    ]
    print(f"[{tag}] canvas-building -> {output_dir}")
    subprocess.run(cmd, cwd=cwm, check=True)


def _combine_canvases(
    cfg, *, inputs: list[Path], output_dir: Path, motor_bounds: dict, tag: str,
) -> None:
    """Merge per-scene canvas datasets into a single final dataset.

    Trivial pass-through when only one scene was recorded — combine_datasets
    still rewrites motor-strip normalization to the bounds we pass, which
    matters for downstream evaluation parity.
    """
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    if output_dir.exists():
        print(f"[{tag}] wiping stale output dir: {output_dir}")
        shutil.rmtree(output_dir)
    cmd = [
        python_exe,
        "combine_datasets.py",
        "--inputs", *[str(p) for p in inputs],
        "--output", str(output_dir),
        "--motor-bounds-json", json.dumps(motor_bounds),
    ]
    print(f"[{tag}] combining {len(inputs)} scene(s) -> {output_dir}")
    subprocess.run(cmd, cwd=cwm, check=True)


# --------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Config YAML (e.g. configs/pan_focus_1b.yaml).")
    p.add_argument("--num-scenes", type=int, default=5,
                   help="How many distinct scenes to record (default 5; "
                        "see module docstring for sizing rationale).")
    p.add_argument("--episodes-per-scene", type=int, default=10,
                   help="Episodes per joint per scene (default 10). "
                        "5 scenes x 10 ep/joint x 2 joints ~ 12 min of "
                        "operator time and produces a val set roughly "
                        "67%% larger than the single-scene baseline.")
    p.add_argument("--shoulder-min", type=float, default=-60.0)
    p.add_argument("--shoulder-max", type=float, default=60.0)
    p.add_argument("--elbow-min", type=float, default=50.0)
    p.add_argument("--elbow-max", type=float, default=90.0)
    p.add_argument("--start-scene", type=int, default=1,
                   help="Resume at this scene index (1-indexed). Earlier "
                        "scene LeRobot recordings must already exist in the "
                        "HF cache under locked_val/<joint>_scene<N>-*. "
                        "Default 1 (record everything from scratch).")
    p.add_argument("--shoulder-only", action="store_true",
                   help="Record only shoulder_pan; skip elbow_flex entirely.")
    p.add_argument("--skip-shoulder", action="store_true",
                   help="Skip shoulder recording phase (use after the "
                        "shoulder canvases were already built).")
    p.add_argument("--skip-elbow", action="store_true",
                   help="Skip elbow recording phase (use after the elbow "
                        "canvases were already built).")
    p.add_argument("--keep-per-scene-builds", action="store_true",
                   help="Don't delete the per-scene canvas dirs after "
                        "merging (useful for debugging / inspection).")
    p.add_argument("--seed-wide-verify", action="store_true",
                   help="Seed the wide-verify corpus instead of recording "
                        "a locked-val. Per-scene canvases are NOT combined; "
                        "each one is appended to the registry's wide-verify "
                        "accumulator so the lifelong learner has a non-empty "
                        "held-out corpus before its first cycle. The seed "
                        "canvases live under runs/<exp>/wide_verify_canvas/ "
                        "and the registry's wide_verify_canvas_dirs is "
                        "populated. Cannot be combined with --keep-per-scene-builds "
                        "(seed mode keeps everything by definition).")
    args = p.parse_args()

    if args.num_scenes < 1:
        sys.exit("--num-scenes must be >= 1")
    if args.episodes_per_scene < 1:
        sys.exit("--episodes-per-scene must be >= 1")
    if args.start_scene < 1 or args.start_scene > args.num_scenes:
        sys.exit(f"--start-scene must be in [1, {args.num_scenes}]")
    if args.shoulder_only and args.skip_shoulder:
        sys.exit("--shoulder-only and --skip-shoulder are contradictory.")

    cfg = load_config(args.config)

    mb_ns = getattr(getattr(cfg, "training", None), "motor_bounds", None)
    if mb_ns is None:
        sys.exit("ERROR: cfg.training.motor_bounds is required.")
    motor_bounds = {k: list(v) for k, v in vars(mb_ns).items() if v is not None}

    do_shoulder = not args.skip_shoulder
    do_elbow = not (args.skip_elbow or args.shoulder_only)
    # Auto-skip elbow when the config doesn't have an elbow locked-val
    # path (e.g. pan_only_1b, where elbow_flex is not in the experiment).
    # Without this, --shoulder-only would have to be passed every time.
    elbow_path_cfg = getattr(cfg.paths, "locked_val_elbow", None)
    if do_elbow and not elbow_path_cfg:
        print("\n[note] cfg.paths.locked_val_elbow is null/missing — "
              "auto-skipping elbow recording. Pass --skip-elbow to silence "
              "this message, or set locked_val_elbow in the config.")
        do_elbow = False
    if not do_shoulder and not do_elbow:
        sys.exit("Nothing to record (both joints skipped).")

    # Per-scene LeRobot recordings, indexed by scene_idx -> Path.
    shoulder_lerobot: dict[int, Path] = {}
    elbow_lerobot: dict[int, Path] = {}

    print("\nMulti-scene locked-val recording")
    print(f"  config:             {args.config}")
    print(f"  num scenes:         {args.num_scenes}")
    print(f"  episodes/scene:     {args.episodes_per_scene}")
    print(f"  start scene:        {args.start_scene}")
    print(f"  joints:             "
          f"{'shoulder_pan' if do_shoulder else ''}"
          f"{' + ' if do_shoulder and do_elbow else ''}"
          f"{'elbow_flex' if do_elbow else ''}")
    print(f"  shoulder out:       {cfg.paths.locked_val_shoulder}")
    if do_elbow:
        print(f"  elbow out:          {cfg.paths.locked_val_elbow}")

    # ---------------------------------------------------- record per scene
    # In --seed-wide-verify mode, force scene 1 into "rearrange" mode so
    # the operator perturbs the trained scene rather than recording the
    # held-out corpus on the same physical setup the model has already
    # seen. Locked-val mode keeps the default ("initial" for scene 1)
    # because locked-val is typically recorded BEFORE training starts —
    # there's no trained scene to rearrange away from yet.
    for scene_idx in range(args.start_scene, args.num_scenes + 1):
        if args.seed_wide_verify:
            _prompt_for_scene(scene_idx, args.num_scenes, mode="rearrange")
        else:
            _prompt_for_scene(scene_idx, args.num_scenes)

        if do_shoulder:
            shoulder_lerobot[scene_idx] = _record_one_joint(
                cfg,
                joint_name="shoulder_pan.pos",
                joint_min=args.shoulder_min, joint_max=args.shoulder_max,
                num_episodes=args.episodes_per_scene,
                repo_id_prefix=f"locked_val/shoulder_scene{scene_idx}",
                tag=f"SHOULDER s{scene_idx}",
            )

        if do_elbow:
            elbow_lerobot[scene_idx] = _record_one_joint(
                cfg,
                joint_name="elbow_flex.pos",
                joint_min=args.elbow_min, joint_max=args.elbow_max,
                num_episodes=args.episodes_per_scene,
                repo_id_prefix=f"locked_val/elbow_scene{scene_idx}",
                tag=f"ELBOW s{scene_idx}",
            )

    # If the user resumed mid-way, the early scenes' LeRobot data must
    # still exist in the HF cache for canvas-build to find them.
    if args.start_scene > 1:
        print(f"\n[resume] expecting earlier scenes "
              f"(1..{args.start_scene - 1}) to already exist in the HF "
              f"cache. Skipping their re-record but they MUST be present "
              f"for canvas-build below to succeed.")
        for prior in range(1, args.start_scene):
            if do_shoulder and prior not in shoulder_lerobot:
                cache = (Path.home() / ".cache" / "huggingface" / "lerobot"
                         / f"locked_val/shoulder_scene{prior}")
                # Match the timestamped subdir collect_batch_continuous
                # creates — pick the newest if multiple exist.
                matches = sorted(cache.parent.glob(f"shoulder_scene{prior}-*"))
                if not matches:
                    sys.exit(f"--start-scene={args.start_scene} but no "
                             f"prior recording found for shoulder scene "
                             f"{prior} under {cache.parent}/")
                shoulder_lerobot[prior] = matches[-1]
                print(f"  found prior shoulder scene {prior}: "
                      f"{matches[-1]}")
            if do_elbow and prior not in elbow_lerobot:
                cache = (Path.home() / ".cache" / "huggingface" / "lerobot"
                         / f"locked_val/elbow_scene{prior}")
                matches = sorted(cache.parent.glob(f"elbow_scene{prior}-*"))
                if not matches:
                    sys.exit(f"--start-scene={args.start_scene} but no "
                             f"prior recording found for elbow scene "
                             f"{prior} under {cache.parent}/")
                elbow_lerobot[prior] = matches[-1]
                print(f"  found prior elbow scene {prior}: "
                      f"{matches[-1]}")

    # ------------------------------------------------- per-scene canvas build
    # In locked-val mode, per-scene canvases go under a temp build dir
    # since they'll be combined into the final locked_val_* paths. In
    # seed-wide-verify mode they're permanent — they go into the
    # canonical wide_verify_canvas/ directory the live learner also
    # writes to, and each one gets registered in the registry's
    # wide-verify accumulator.
    runs_dir = Path(cfg.paths.runs_dir)
    if args.seed_wide_verify:
        build_root = runs_dir / "wide_verify_canvas"
        path_prefix = "seed"
    else:
        build_root = runs_dir / "locked_val_build"
        path_prefix = ""
    build_root.mkdir(parents=True, exist_ok=True)

    shoulder_canvas_dirs: list[Path] = []
    elbow_canvas_dirs: list[Path] = []
    for scene_idx in sorted(set(shoulder_lerobot) | set(elbow_lerobot)):
        if scene_idx in shoulder_lerobot:
            out = build_root / (
                f"{path_prefix}_shoulder_scene{scene_idx}".lstrip("_")
            )
            _canvas_build(
                cfg,
                lerobot_path=shoulder_lerobot[scene_idx],
                output_dir=out, motor_bounds=motor_bounds,
                tag=f"SHOULDER s{scene_idx}",
            )
            shoulder_canvas_dirs.append(out)
        if scene_idx in elbow_lerobot:
            out = build_root / (
                f"{path_prefix}_elbow_scene{scene_idx}".lstrip("_")
            )
            _canvas_build(
                cfg,
                lerobot_path=elbow_lerobot[scene_idx],
                output_dir=out, motor_bounds=motor_bounds,
                tag=f"ELBOW s{scene_idx}",
            )
            elbow_canvas_dirs.append(out)

    if args.seed_wide_verify:
        # ----------------------------- register seed canvases in registry
        # Each scene's canvas dir becomes a separate held-out entry.
        # The Registry's leakage invariant rejects any path that would
        # also be in `accumulated_canvas_dirs` — but seed dirs are
        # produced before any training has run so the invariant is
        # trivially satisfied.
        registry = Registry(cfg.paths.registry_file)
        all_dirs = list(zip(shoulder_canvas_dirs, ["shoulder"] * len(shoulder_canvas_dirs))) \
                 + list(zip(elbow_canvas_dirs, ["elbow"] * len(elbow_canvas_dirs)))
        for canvas_dir, joint in all_dirs:
            registry.append_wide_verify_dir(
                canvas_dir,
                n_probes=args.episodes_per_scene,
                scenes=args.num_scenes,
                cycle=-1,  # -1 = pre-cycle seed
            )
            print(f"[seed] registered {joint}: {canvas_dir}")
        print(f"\n[seed-wide-verify] Done. Registered "
              f"{len(all_dirs)} canvas dirs in "
              f"{cfg.paths.registry_file}")
        print(f"  These canvases will be evaluated against every "
              f"future retrain via wide_verify_mse.")
    else:
        # --------------------------------------------- combine to final out
        if shoulder_canvas_dirs:
            _combine_canvases(
                cfg, inputs=shoulder_canvas_dirs,
                output_dir=Path(cfg.paths.locked_val_shoulder),
                motor_bounds=motor_bounds, tag="SHOULDER",
            )
        if elbow_canvas_dirs:
            _combine_canvases(
                cfg, inputs=elbow_canvas_dirs,
                output_dir=Path(cfg.paths.locked_val_elbow),
                motor_bounds=motor_bounds, tag="ELBOW",
            )

        # ------------------------------------------------- cleanup + done
        if not args.keep_per_scene_builds:
            try:
                shutil.rmtree(build_root)
                print(f"\n[cleanup] removed per-scene build dir: {build_root}")
            except OSError as e:
                print(f"\n[cleanup] could not remove {build_root}: {e}")

        print("\nDone.")
        if shoulder_canvas_dirs:
            print(f"  shoulder locked-val canvas dataset: "
                  f"{cfg.paths.locked_val_shoulder} "
                  f"({len(shoulder_canvas_dirs)} scenes merged)")
        if elbow_canvas_dirs:
            print(f"  elbow locked-val canvas dataset:    "
                  f"{cfg.paths.locked_val_elbow} "
                  f"({len(elbow_canvas_dirs)} scenes merged)")
    print(f"\nReady to start the learner:")
    print(f"  {cfg.paths.python} -m learner --config {args.config}")


if __name__ == "__main__":
    main()
