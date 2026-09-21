"""Versioned raw episodes: observation at t, applied target during [t,t+dt]."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "piper-titration-v1"
ORDER = [f"{arm}_{joint}" for arm in ("holder", "squeezer") for joint in (*[f"joint{i}" for i in range(1, 7)], "width")]


class Recorder:
    def __init__(self, destination, metadata):
        self.path = Path(destination)
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema": SCHEMA, "order": ORDER, "units": "radians; gripper total width in metres", **metadata}
        self.rows = []

    def add(self, observation, action, timestamp):
        i = len(self.rows)
        # Per-frame compressed files bound RAM usage, including failed episodes.
        np.savez_compressed(self.path / f"{i:06d}.npz", state=observation["state"], action=action, image=observation["image"], wrist_image=observation["wrist_image"], timestamp=np.float64(timestamp))
        self.rows.append(i)

    def finish(self, report):
        self.metadata.update(report=report, frames=len(self.rows))
        (self.path / "episode.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")


def read(path):
    path = Path(path)
    metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if metadata["schema"] != SCHEMA or metadata["order"] != ORDER:
        raise ValueError(f"Unsupported episode schema/order: {path}")
    frames = sorted(path.glob("*.npz"))
    if len(frames) != metadata["frames"] or not frames:
        raise ValueError(f"Incomplete or empty episode: {path}")
    def iterator():
        previous = None
        for frame in frames:
            with np.load(frame, allow_pickle=False) as f:
                row = {key: f[key] for key in f.files}
            for key in ("state", "action"):
                if row[key].shape != (14,) or not np.isfinite(row[key]).all():
                    raise ValueError(f"Invalid {key}: {frame}")
            for key in ("image", "wrist_image"):
                if row[key].ndim != 3 or row[key].shape[-1] != 3 or row[key].dtype != np.uint8:
                    raise ValueError(f"Expected HWC uint8 RGB {key}: {frame}")
            t = float(row["timestamp"])
            if not np.isfinite(t) or (previous is not None and not np.isclose(t-previous, 1/metadata["fps"], atol=1e-5)):
                raise ValueError(f"Invalid or irregular timestamp: {frame}")
            previous = t
            yield row
    return metadata, iterator()


def partition(paths, seed=42, test_fraction=.2):
    if len(paths) < 2:
        raise ValueError("At least two complete episodes required for train/test split")
    paths = sorted(paths, key=lambda p: hashlib.sha256(f"{seed}:{Path(p).parent.name}/{Path(p).name}".encode()).hexdigest())
    n = max(1, min(len(paths)-1, round(len(paths)*test_fraction)))
    return {"test": paths[:n], "train": paths[n:]}


def convert(raw, repo_id, seed=42):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    paths = []
    for p in sorted(Path(raw).rglob("episode.json")):
        meta, rows = read(p.parent)
        if meta.get("report", {}).get("success"):
            paths.append(p.parent)
    split = partition(paths, seed)
    manifest = {}
    for name, episodes in split.items():
        rid = f"{repo_id}_{name}"
        if (HF_LEROBOT_HOME / rid).exists():
            raise FileExistsError(f"Refusing to overwrite {HF_LEROBOT_HOME / rid}")
        meta, rows = read(episodes[0])
        first = next(rows)
        features = {key: {"dtype": "image", "shape": first[key].shape, "names": ["height", "width", "channel"]} for key in ("image", "wrist_image")}
        features.update({key: {"dtype": "float32", "shape": (14,), "names": ORDER} for key in ("state", "actions")})
        dataset = LeRobotDataset.create(repo_id=rid, robot_type="dual_piper", fps=meta["fps"], features=features, use_videos=False, image_writer_threads=2, image_writer_processes=0)
        for episode in episodes:
            m, rows = read(episode)
            if m["fps"] != meta["fps"]:
                raise ValueError("Mixed frame rates are unsupported")
            for row in rows:
                dataset.add_frame({"state": row["state"].astype(np.float32), "actions": row["action"].astype(np.float32), "image": row["image"], "wrist_image": row["wrist_image"], "task": m["prompt"]})
            dataset.save_episode()
        manifest[rid] = [str(p.resolve()) for p in episodes]
    output = Path(raw) / "split.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
