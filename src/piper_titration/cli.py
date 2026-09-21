import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Dual PiPER titration simulation")
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
    commands.add_parser("train").add_argument("--steps", type=int)
    sub = commands.add_parser("serve")
    sub.add_argument("--checkpoint", required=True)
    sub.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from .config import load
    cfg = load(args.config)
    if args.command == "doctor":
        from .diagnostics import inspect
        result = inspect(**{k: cfg["paths"][k] for k in ("sim_python", "train_python", "openpi_root", "checkpoint", "piper_root")}, render=args.render)
        print(json.dumps(result, indent=2))
        return int(any(not r["ok"] for r in result.values()))
    if args.command == "build":
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
