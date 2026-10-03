"""Two independent fixed-site policies. Run on NAS; training executes on the SSH node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import train_remote_3090 as remote


def task_entries(profile, task):
    entries = json.loads(Path(profile).read_text(encoding="utf-8"))["tasks"]
    names = ["beaker_move", "tube_pour"] if task == "both" else [task]
    return [(name, entries[name]) for name in names]


def validate_task_receipt(receipt, accepted):
    episodes = receipt["splits"]["train"]["episodes"]
    if not episodes or any(e.get("task_id") not in accepted for e in episodes):
        raise ValueError("Receipt contains another task or has no episodes; refusing mixed-task training")


def validate_isolation(configs):
    for field in ("repo_id", "output", "receipt"):
        values = [c[field] for c in configs]
        if len(values) != len(set(values)):
            raise ValueError(f"Independent skills must use distinct {field}")


def worker(args):
    root = Path.cwd()
    sys.path.insert(0, str(root/"src"))
    # SSH wrapper's old beaker defaults must not override per-task isolation.
    env = dict(os.environ)
    for key in ("PIPER_CONFIG", "PIPER_OUTPUT", "PIPER_DATASET_RECEIPT"):
        env.pop(key, None)
        os.environ.pop(key, None)
    from piper_titration.config import load
    from piper_titration.capture_dataset import validate_receipt
    from piper_titration.training import attach, validate_device_layout
    selected = task_entries(args.profile,args.task)
    layouts = []
    for _, spec in selected:
        cfg = load(root/spec["config"])
        layouts.append({"repo_id":cfg["training"]["repo_id"],
                        "output":str(Path(cfg["paths"]["output"]).resolve()),
                        "receipt":str((root/spec["receipt"]).resolve())})
    validate_isolation(layouts)
    import jax
    validate_device_layout(args.batch_size,args.fsdp_devices,jax.device_count())
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("No GPU available")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root/"outputs/two_skills"/run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    status = {"scope":"two_independent_fixed_site_policies", "tasks":{}}
    def save():
        (run_dir/"run.json").write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding="utf-8")
    save()
    for name, spec in selected:
        record = status["tasks"][name] = {"status":"running"}
        save()
        try:
            cfg = load(root/spec["config"])
            cfg["paths"]["dataset_receipt"] = str((root/spec["receipt"]).resolve())
            cfg["training"].update(batch_size=args.batch_size, fsdp_devices=args.fsdp_devices)
            config_path = run_dir/(name+".json")
            config_path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
            record["config"] = str(config_path)
            invoke = [sys.executable,"-m","piper_titration","--config",str(config_path)]
            def call(*command, check=True):
                print(shlex.join(invoke+list(command)),flush=True)
                return subprocess.run(invoke+list(command),env=env,check=check).returncode
            receipt_path = Path(cfg["paths"]["dataset_receipt"])
            prepare = args.stage in ("prepare","all")
            if not receipt_path.exists():
                if not prepare:
                    raise FileNotFoundError(f"Run prepare first: {receipt_path}")
                manifest = (root/spec["manifest"]).resolve()
                entries = json.loads(manifest.read_text(encoding="utf-8"))["episodes"]
                if not entries or any(e.get("task_id") not in spec["accepted_task_ids"] for e in entries):
                    raise ValueError(f"Manifest must contain only {name}")
                rid = cfg["training"]["repo_id"]
                if not rid.endswith("_train"):
                    raise ValueError("Training repo_id must end in _train")
                call("convert-captures",str(manifest),"--train-only","--repo-id",rid[:-6])
            # Validate existing data without forcing raw manifests or reconversion.
            attach(cfg)
            receipt = validate_receipt(cfg)
            validate_task_receipt(receipt,set(spec["accepted_task_ids"]))
            record["repo_id"] = cfg["training"]["repo_id"]
            record["receipt"] = str(receipt_path)
            if prepare:
                print("Checking existing normalization; a new dataset has no norm audit yet.",flush=True)
                if call("norm-check",check=False):
                    print("Normalization is missing or invalid; computing fresh statistics before training.",flush=True)
                    call("norms")
            call("norm-check")
            modes = ["smoke","train"] if args.stage == "all" else ([args.stage] if args.stage in ("smoke","train") else [])
            for mode in modes:
                steps = 1 if mode == "smoke" else args.steps
                experiment = f"{name}_{mode}_{run_id}"
                call("train","--steps",str(steps),"--experiment",experiment)
                step = Path(cfg["paths"]["output"])/"checkpoints/pi05_piper_chemical_lora"/experiment/str(steps-1)
                if not (step/"params").is_dir() or not (step/"assets").is_dir():
                    raise RuntimeError(f"Training exited but checkpoint incomplete: {step}")
                record[mode+"_checkpoint"] = str(step)
                save()
            record["status"] = "completed"
            save()
        except Exception as exc:
            record.update(status="failed",error=str(exc))
            save()
            raise
    print(f"Run records and task checkpoint selection: {run_dir/'run.json'}",flush=True)
    return 0


def main():
    # Reuse the tested key-based SSH options without duplicating connection logic.
    p = remote.parser()
    p.description = __doc__
    # Positional mode is still supplied internally as doctor, not user-facing.
    p.add_argument("--stage",choices=("prepare","smoke","train","all"),default="smoke")
    p.add_argument("--task",choices=("both","beaker_move","beaker_move_new","tube_pour"),default="both")
    p.add_argument("--profile",default="configs/two_skills.json")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = p.parse_args(["doctor"]+sys.argv[1:])
    if args.worker:
        return worker(args)
    command = ["scripts/train_two_skills.py","--worker","--stage",args.stage,"--task",args.task,
               "--profile",args.profile,"--steps",str(args.steps),"--batch-size",str(args.batch_size),
               "--fsdp-devices",str(args.fsdp_devices)]
    if args.steps < 1 or args.batch_size < 1 or args.fsdp_devices < 1:
        p.error("steps, batch-size and fsdp-devices must be positive")
    try:
        ssh,script = remote.build(args,worker_command=command)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(shlex.join(ssh)); print(script)
        return 0
    return subprocess.run(ssh,input=script,text=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
