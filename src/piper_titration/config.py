from pathlib import Path
import os
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(path=None):
    path = Path(path or ROOT / "configs/chemical_server.json")
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".json":
            cfg = json.load(f)
        else:
            import yaml
            cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        value = os.environ.get("PIPER_" + key.upper(), value)
        if value == "@current":
            value = sys.executable
        p = Path(value).expanduser()
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    if "robot" in cfg:
        from .single_arm import validate_contract
        validate_contract(cfg["robot"])
    else:
        s = cfg["simulation"]
        if s["control_hz"] <= 0 or s["timestep"] <= 0:
            raise ValueError("control_hz and timestep must be positive")
        ratio = 1 / s["control_hz"] / s["timestep"]
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("control period must be an integer multiple of timestep")
        for key in ("initial_volume_ml", "drop_volume_ml", "displacement_ml", "relaxation_seconds", "release_seconds"):
            if cfg["fluid"][key] <= 0:
                raise ValueError(f"fluid.{key} must be positive")
    if cfg["training"]["batch_size"] < 1 or cfg["training"]["steps"] < 1:
        raise ValueError("training batch_size and steps must be positive")
    return cfg
