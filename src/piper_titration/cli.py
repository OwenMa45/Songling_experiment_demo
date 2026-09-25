import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Single-arm PiPER chemical imitation learning")
    p.add_argument("--config")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor").add_argument("--render", action="store_true")
    commands.add_parser("build")
    for name in ("demo", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True)
        sub.add_argument("--randomized", action="store_true")
        sub.add_argument("--record", action="store_true")
        sub.add_argument("--policy-uri")
        if name == "demo":
            sub.add_argument("--target", type=int, default=3)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--video", action="store_true")
        else:
            sub.add_argument("--episodes", type=int, default=100)
    commands.add_parser("replay").add_argument("episode")
    sub = commands.add_parser("convert")
    sub.add_argument("raw")
    sub.add_argument("--repo-id", default="local/piper_titration")
    commands.add_parser("norms")
    for name in ("capture-plan", "convert-captures"):
        sub = commands.add_parser(name)
        sub.add_argument("manifest")
        sub.add_argument("--train-only", action="store_true", help="Explicit fixed-scene training without validation/test; no generalization claim")
        if name == "convert-captures":
            sub.add_argument("--repo-id", default="local/piper_chemical")
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int)
    train.add_argument("--experiment", help="New run name; use separate names for smoke and full training")
    sub = commands.add_parser("serve")
    sub.add_argument("--checkpoint", required=True)
    sub.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from .config import load
    cfg = load(args.config)
    if args.command == "train" and args.experiment:
        if any(c in args.experiment for c in "/\\") or args.experiment in (".",".."):
            p.error("experiment must be a directory name, not a path")
        cfg["training"]["experiment"] = args.experiment
    if "robot" in cfg and args.command in ("build","demo","evaluate","replay","convert"):
        p.error("This is a legacy dual-arm command. For real single-arm data use capture-plan/convert-captures. Legacy simulation requires --config configs/server.yaml.")
    if args.command == "doctor":
        from .diagnostics import inspect
        result = inspect(**{k: cfg["paths"][k] for k in ("sim_python", "train_python", "openpi_root", "checkpoint", "piper_root")}, render=args.render)
        print(json.dumps(result, indent=2))
        return int(any(not r["ok"] for r in result.values()))
    if args.command in ("capture-plan", "convert-captures"):
        if "robot" not in cfg:
            p.error("Capture commands require configs/chemical_server.json")
        from .capture_dataset import plan_dataset, convert_captures
        if args.command == "capture-plan":
            print(json.dumps(plan_dataset(args.manifest,cfg["training"]["seed"],train_only=args.train_only),indent=2,ensure_ascii=False))
        else:
            print(convert_captures(cfg,args.manifest,args.repo_id,train_only=args.train_only))
    elif args.command == "build":
        from pathlib import Path
        from .scene import build
        path = Path(cfg["paths"]["output"])/"generated/dual_piper.xml"
        build(cfg, path)
        print(path)
    elif args.command in ("demo", "evaluate"):
        from .runner import run, evaluate
        options = {k: getattr(args, k) for k in ("randomized", "record", "policy_uri")}
        if args.command == "demo":
            result = run(cfg, args.output, target=args.target, seed=args.seed, video=args.video, **options)
        else:
            result = evaluate(cfg, args.output, episodes=args.episodes, **options)
        print(json.dumps(result, indent=2))
        return int(not (result["success"] if args.command == "demo" else result["acceptance_passed"]))
    elif args.command == "replay":
        from .runner import replay
        print(json.dumps(replay(args.episode), indent=2))
    elif args.command == "convert":
        from .training import attach
        attach(cfg)
        from .episodes import convert
        print(json.dumps(convert(args.raw, args.repo_id), indent=2))
    else:
        from .training import execute
        execute(cfg, args.command, steps=getattr(args, "steps", None), checkpoint=getattr(args, "checkpoint", None), port=getattr(args, "port", 8000))
    return 0
