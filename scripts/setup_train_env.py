"""Create an isolated Linux training environment from the pinned openpi lockfile."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from piper_titration.diagnostics import PINS


def run(args, **kwargs):
    print('+ '+shlex.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), check=True, **kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root',type=Path,default=Path(os.getenv('PIPER_OPENPI_ROOT','/mnt/cpfs/users/mrq/emboddied/openpi')))
    parser.add_argument('--env',type=Path,default=ROOT/'.venv-train')
    parser.add_argument('--skip-gpu-check',action='store_true',help='Install and import-check without requiring a visible GPU')
    args=parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Run this training setup on the Linux NAS, not Windows.')
    if not shutil.which('uv') or not shutil.which('git'):
        parser.error('Install uv and git first; see docs/training_environment.md')
    source=args.openpi_root.expanduser().resolve()
    target=args.env.expanduser().resolve()
    revision=run(['git','-C',source,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    if revision != PINS['openpi']:
        parser.error(f'openpi must match {PINS["openpi"]}; select third_party/openpi instead of resetting your existing checkout.')
    if not (source/'uv.lock').is_file():
        parser.error('Missing openpi uv.lock; initialize the pinned checkout first.')
    if target == ROOT or target == source or target in ROOT.parents or target in source.parents:
        parser.error('Environment must be a separate directory, not a project or its ancestor.')
    marker=target/'.piper-training-env.json'
    expected={'project':str(ROOT),'openpi_root':str(source),'revision':revision}
    if target.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            parser.error('Refusing to synchronize an existing unmanaged directory. Choose a new --env path.')
    else:
        target.mkdir(parents=True)
        marker.write_text(json.dumps(expected,indent=2))
    env=dict(os.environ,UV_PROJECT_ENVIRONMENT=str(target),GIT_LFS_SKIP_SMUDGE='1')
    env.pop('VIRTUAL_ENV',None)
    # --locked checks consistency and never rewrites the upstream lockfile.
    run(['uv','sync','--project',source,'--locked','--python','3.11'],env=env)
    python=target/'bin/python'
    # All runtime dependencies already come from openpi's lock; do not re-resolve them.
    run(['uv','pip','install','--python',python,'--no-deps','-e',ROOT],env=env)
    run(['uv','pip','check','--python',python],env=env)
    run([python,'-c','import numpy, PIL, yaml, jax, flax, orbax.checkpoint, lerobot, openpi, openpi_client, piper_titration; print("Training imports OK; JAX",jax.__version__)'],env=env)
    if not args.skip_gpu_check:
        run([python,'-c','import jax; devices=jax.devices(); print(devices); assert any(d.platform=="gpu" for d in devices), "No JAX GPU detected: check GPU allocation and NVIDIA driver"'],env=env)
    activation=target/'piper_env.sh'
    activation.write_text(
        '# Source this file before running project commands.\n'
        f'source {shlex.quote(str(target/"bin/activate"))}\n'
        f'export PIPER_TRAIN_PYTHON={shlex.quote(str(python))}\n'
        f'export PIPER_OPENPI_ROOT={shlex.quote(str(source))}\n'
        f'export PYTHONPATH={shlex.quote(str(ROOT/"src"))}${{PYTHONPATH:+:$PYTHONPATH}}\n')
    print(f'Environment ready. Run: source {shlex.quote(str(activation))}')
    print('Weights, datasets and actual training are not downloaded or started by this script.')


if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError,OSError,ValueError) as exc:
        print(f'Setup failed: {exc}. Environment retained for diagnosis; fix the issue and rerun.',file=sys.stderr)
        raise SystemExit(1)
