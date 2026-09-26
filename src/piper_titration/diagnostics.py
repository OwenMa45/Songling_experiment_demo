"""Read-only checks; no installers, downloads, generated scenes or checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PINS = {"openpi": "215abfb217dbac7d5f1273282331b9b1866c0479", "piper_ros": "ac41fcbcdda598f01b51cf6175ed9a24d0dacadc"}
BASE = "/mnt/cpfs/users/mrq/emboddied"


def command(args, timeout=40, env=None):
    try:
        environ = dict(os.environ if env is None else env, PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=environ)
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def inspect(sim_python, train_python, openpi_root, checkpoint, piper_root, render=False, training_only=False):
    results = {}
    for name, path in {"sim_python": sim_python, "train_python": train_python, "openpi_root": openpi_root, "checkpoint": checkpoint, "piper_root": piper_root}.items():
        results[name] = {"ok": Path(path).exists(), "path": str(path)}
    probe = "import importlib.metadata as m,json; names=%r; out={};\nfor n in names:\n try: out[n]=m.version(n)\n except m.PackageNotFoundError: out[n]=None\nprint(json.dumps(out)); raise SystemExit(any(v is None for v in out.values()))"
    if not training_only:
        results["simulation_packages"] = command([str(sim_python), "-c", probe % ["mujoco", "numpy", "PyYAML", "imageio", "imageio-ffmpeg", "openpi-client"]])
    results["jax_gpu"] = command([str(train_python), "-c", "import jax; d=jax.devices(); print(d); assert any(x.platform=='gpu' for x in d), 'JAX has no GPU device'"])
    results["training_packages"] = command([str(train_python), "-c", probe % ["jax", "flax", "orbax-checkpoint", "lerobot", "openpi", "PyYAML"]])
    results["gpu"] = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    for name, path in (("openpi", openpi_root), ("piper_ros", piper_root)):
        if training_only and name == "piper_ros":
            continue
        r = command(["git", "-C", str(path), "rev-parse", "HEAD"])
        r["expected"] = PINS[name]
        r["ok"] = r["ok"] and r.get("stdout") == PINS[name]
        results[name + "_revision"] = r
    params = Path(checkpoint) / "params"
    results["jax_checkpoint"] = {"ok": params.is_dir() and any(params.iterdir()), "path": str(params), "note": "Structural check only; actual restore is validated by training smoke test."}
    if render:
        code = "import mujoco,numpy as np; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"sphere\" size=\"0.1\"/></worldbody></mujoco>'); d=mujoco.MjData(m); mujoco.mj_forward(m,d); r=mujoco.Renderer(m,64,64); r.update_scene(d); a=r.render(); assert a.shape==(64,64,3); print(a.shape); r.close()"
        environ = dict(os.environ, MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1")
        results["egl_render"] = command([str(sim_python), "-c", code], env=environ)
        model = Path(piper_root) / "src/piper_description/mujoco_model/piper_description.xml"
        results["single_arm_render"] = command([str(sim_python), "-c", "import mujoco,sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_forward(m,d); r=mujoco.Renderer(m,128,128); r.update_scene(d); a=r.render(); assert a.shape==(128,128,3); print(m.nq,m.nu,a.shape); r.close()", str(model)], env=environ)
    if training_only:
        results.pop("sim_python",None)
        results.pop("piper_root",None)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-python", default=os.getenv("PIPER_SIM_PYTHON", BASE+"/mujoco/bin/python"))
    parser.add_argument("--train-python", default=os.getenv("PIPER_TRAIN_PYTHON", BASE+"/openpi/.venv/bin/python"))
    parser.add_argument("--openpi-root", default=os.getenv("PIPER_OPENPI_ROOT", BASE+"/openpi"))
    parser.add_argument("--checkpoint", default=os.getenv("PIPER_CHECKPOINT", BASE+"/openpi/checkpoints/pi05_base"))
    parser.add_argument("--piper-root", default=str(Path(__file__).resolve().parents[2]/"third_party/piper_ros"))
    parser.add_argument("--render", action="store_true")
    result = inspect(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(any(not r["ok"] for r in result.values()))


if __name__ == "__main__":
    sys.exit(main())
