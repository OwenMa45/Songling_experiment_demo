"""Launch project commands on the 3090 node via key-based SSH (standard library only)."""
import argparse
from datetime import datetime, timezone
from pathlib import PurePosixPath
import shlex
import subprocess

BASE = "/mnt/nas/share/home/lzk/mrq/robot"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("doctor", "norms", "smoke", "train"))
    p.add_argument("--host", default="10.71.106.251")
    p.add_argument("--user", default="lzk")
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--identity", default="/home/lzk/.ssh/id_ed25519", help="Private key path on the NAS launching this script; never copied")
    p.add_argument("--project", default=BASE + "/Songling_experiment_demo")
    p.add_argument("--openpi", default=BASE + "/third_party/openpi")
    p.add_argument("--checkpoint", default=BASE + "/openpi_data/openpi-assets/checkpoints/pi05_base")
    p.add_argument("--conda", default="/home/lzk/miniconda3/bin/conda")
    p.add_argument("--env", default="songling")
    p.add_argument("--python", help="Remote environment Python; overrides --conda/--env")
    p.add_argument("--config", default="configs/beaker_demo.json")
    p.add_argument("--receipt", default="outputs/chemical/dataset_receipt.json")
    p.add_argument("--lerobot", default="data/lerobot")
    p.add_argument("--output", default="outputs/beaker_demo", help="Keep existing norms assets accessible")
    p.add_argument("--cache", default=BASE + "/openpi_data")
    p.add_argument("--hf-cache", default=BASE + "/cache/huggingface")
    p.add_argument("--gpu-list", default="0,1,2,3")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fsdp-devices", type=int, default=4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--experiment", help="Defaults to a new UTC timestamped experiment")
    p.add_argument("--memory-fraction", type=float, default=.9)
    p.add_argument("--dry-run", action="store_true", help="Print SSH command and remote script without connecting")
    return p


def build(args):
    ids = args.gpu_list.split(",")
    if not ids or len(set(ids)) != len(ids) or any(i not in ("0", "1", "2", "3") for i in ids):
        raise ValueError("3090 GPU list must contain unique IDs from 0,1,2,3")
    if not 0 < args.memory_fraction < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid memory fraction or SSH port")
    if args.mode in ("smoke", "train"):
        if args.batch_size < 1 or args.batch_size % len(ids) or args.fsdp_devices < 1 or len(ids) % args.fsdp_devices:
            raise ValueError("Batch must divide evenly across visible GPUs; FSDP must divide GPU count")
        if args.steps < 1:
            raise ValueError("steps must be positive")
    if not args.host or args.host.startswith("-") or not args.user or args.user.startswith("-") or any(c.isspace() for c in args.host+args.user):
        raise ValueError("Invalid SSH host/user")
    if not PurePosixPath(args.project).is_absolute():
        raise ValueError("Remote project must be an absolute Linux path")
    def remote(value):
        p = PurePosixPath(value)
        return str(p if p.is_absolute() else PurePosixPath(args.project)/p)
    experiment = args.experiment or f"beaker_{args.mode}_3090_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    if not experiment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in experiment):
        raise ValueError("Experiment must use letters, digits, _ or -")
    env = {
        "PIPER_CONFIG":remote(args.config), "PIPER_DATASET_RECEIPT":remote(args.receipt),
        "PIPER_OPENPI_ROOT":args.openpi, "PIPER_CHECKPOINT":args.checkpoint,
        "PIPER_OUTPUT":remote(args.output), "HF_LEROBOT_HOME":remote(args.lerobot),
        "OPENPI_DATA_HOME":args.cache, "HF_HOME":args.hf_cache,
        "CUDA_VISIBLE_DEVICES":args.gpu_list,
        "XLA_PYTHON_CLIENT_MEM_FRACTION":str(args.memory_fraction),
        "XLA_PYTHON_CLIENT_PREALLOCATE":"true",
        "PYTHONPATH":remote("src"),
    }
    q = shlex.quote
    runner = [args.python] if args.python else [args.conda, "run", "--no-capture-output", "-n", args.env, "python"]
    invoke = runner+["-m", "piper_titration", "--config", remote(args.config)]
    script = ["set -euo pipefail", "cd -- " + q(args.project)]
    script += [f"export {k}={q(v)}" for k,v in env.items()]
    log = remote(f"outputs/remote_logs/{experiment}.log")
    script += ["mkdir -p -- " + q(str(PurePosixPath(log).parent)),
               "printf 'Remote host: '; hostname",
               "printf 'CUDA_VISIBLE_DEVICES=%s\\n' \"$CUDA_VISIBLE_DEVICES\"",
               "test -f " + q(remote(args.config))]
    command = invoke + (["train", "--steps", str(1 if args.mode=="smoke" else args.steps),
                         "--batch-size", str(args.batch_size), "--fsdp-devices", str(args.fsdp_devices),
                         "--experiment", experiment] if args.mode in ("smoke", "train") else [args.mode])
    if args.mode != "doctor":
        script += ["test -f " + q(remote(args.receipt)), shlex.join(invoke+["doctor"])]
    script += [shlex.join(command)+" 2>&1 | tee -- "+q(log)]
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-p", str(args.port)]
    if args.identity:
        ssh += ["-o", "IdentitiesOnly=yes", "-i", args.identity]
    ssh += [args.user+"@"+args.host, "bash -s"]
    return ssh, "\n".join(script)+"\n"


def main():
    p = parser()
    args = p.parse_args()
    try:
        command, script = build(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(shlex.join(command))
        print(script)
        return 0
    # Pass commands on stdin; paths cannot become remote shell command substitutions.
    try:
        return subprocess.run(command, input=script, text=True).returncode
    except FileNotFoundError:
        p.error("OpenSSH client 'ssh' was not found")


if __name__ == "__main__":
    raise SystemExit(main())
