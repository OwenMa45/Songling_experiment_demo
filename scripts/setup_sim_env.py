"""Create/reuse an isolated simulation venv, install dependencies and test rendering."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    print('+ ' + shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=ROOT/'.venv-sim', help='Environment directory; default: project/.venv-sim')
    parser.add_argument('--python', default=sys.executable, help='Python 3.11 or 3.12 used to create the venv')
    parser.add_argument('--skip-render', action='store_true', help='Check physics only; do not test EGL rendering')
    args = parser.parse_args()
    target = args.env.expanduser().resolve()
    python = target/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if target.exists():
        if not (target/'pyvenv.cfg').is_file() or not python.is_file():
            parser.error(f'Refusing to modify an existing non-venv directory: {target}')
    else:
        # NumPy <2 in this project has wheels for these interpreter versions.
        run([args.python, '-c', 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12"'])
        run([args.python, '-m', 'venv', str(target)])
    run([python, '-c', 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use a Python 3.11 or 3.12 venv"'])
    run([python, '-m', 'pip', 'install', '-e', str(ROOT)+'[simulation]'])
    run([python, '-m', 'pip', 'check'])
    env = dict(os.environ)
    if sys.platform.startswith('linux'):
        env.setdefault('MUJOCO_GL', 'egl')
    code = '''
import mujoco
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print('MuJoCo', mujoco.__version__, 'physics OK')
'''
    if not args.skip_render:
        code += '''
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    pixels = renderer.render()
    assert pixels.shape == (64, 64, 3)
    print('Offscreen rendering OK', pixels.shape)
'''
    run([python, '-c', code], env=env)
    print(f'Environment ready: {target}')
    if os.name == 'nt':
        print(f'$env:PIPER_SIM_PYTHON = "{python}"')
    else:
        print(f'export PIPER_SIM_PYTHON={shlex.quote(str(python))}')
        print(f'export MUJOCO_GL={shlex.quote(env.get("MUJOCO_GL", "egl"))}')
    print('Render check skipped.' if args.skip_render else 'Render check passed.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print('Setup/check failed. Environment retained for diagnosis; no directories deleted. '
              'For EGL errors, check NVIDIA drivers and EGL libraries on the server.', file=sys.stderr)
        raise SystemExit(exc.returncode)
