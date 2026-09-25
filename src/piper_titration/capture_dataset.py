"""Reviewed JSONL/JPEG capture to pinned LeRobot, without action imputation."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from .single_arm import SCHEMA, ORDER

ROOT = Path(__file__).resolve().parents[2]


def audit_episode(path):
    spec = importlib.util.spec_from_file_location("capture_auditor", ROOT/"scripts/audit_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.audit(path, decode_images=True)


def plan_dataset(manifest, seed=42, train_only=False):
    """Preflight EVERY episode before importing LeRobot or writing datasets."""
    manifest = Path(manifest).resolve()
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    entries = selection.get("episodes", [])
    if not entries:
        raise ValueError("Selection manifest must contain reviewed episodes")
    tasks = {t["id"]: t for t in json.loads((ROOT/"configs/chemical_tasks.json").read_text(encoding="utf-8"))["tasks"]}
    seen, content_seen, checked, failures = set(), set(), [], []
    for entry in entries:
        path = (manifest.parent/entry["path"]).resolve()
        if path in seen:
            raise ValueError(f"Duplicate episode path: {path}")
        seen.add(path)
        if not (path/"metadata.json").is_file() or not (path/"samples.jsonl").is_file():
            failures.append({"path":str(path),"blockers":["episode_files_missing"]})
            continue
        if entry.get("task_id") not in tasks or not isinstance(entry.get("group"), str) or not entry["group"].strip():
            raise ValueError("Each episode requires a known task_id and a collection-session/scene group")
        report = audit_episode(path)
        if not report["training_ready"]:
            failures.append({"path": str(path), "blockers": report["blockers"]})
            continue
        digest = report["samples_sha256"]
        if digest in content_seen:
            raise ValueError(f"Duplicate trajectory content: {path}")
        content_seen.add(digest)
        meta = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        if meta.get("state_order") != ORDER:
            raise ValueError(f"Wrong coordinate order: {path}")
        if not {"front", "wrist"}.issubset(meta.get("cameras", {})):
            raise ValueError(f"Current single-arm profile requires front and wrist cameras: {path}")
        checked.append({**entry, "path":str(path), "audit":report, "prompt":tasks[entry["task_id"]]["prompt"]})
    if failures:
        raise ValueError("Rejected captures; no dataset written:\n"+json.dumps(failures,ensure_ascii=False,indent=2))
    if train_only:
        return {"train":checked}
    # Groups never cross partitions, including when a session contains several tasks.
    by_task = defaultdict(set)
    for entry in checked:
        by_task[entry["task_id"]].add(entry["group"])
    if any(len(g)<3 for g in by_task.values()):
        raise ValueError("Each selected task needs at least three independent session/scene groups for train/validation/test")
    groups = sorted({e["group"] for e in checked})
    best = None
    # Deterministic candidate group allocations; preserve group separation first.
    # Small pilot sets may need >10% holdout to represent every task in each split.
    for holdout in range(max(1, round(len(groups)*.1)), len(groups)//3+1):
        for attempt in range(1000):
            ordered = sorted(groups,key=lambda g:hashlib.sha256(f"{seed}:{attempt}:{g}".encode()).hexdigest())
            assignment = {g:("test" if i<holdout else "validation" if i<2*holdout else "train") for i,g in enumerate(ordered)}
            if any({assignment[g] for g in values} != {"train","validation","test"} for values in by_task.values()):
                continue
            score = 0
            for task in by_task:
                counts = {s:sum(e["task_id"]==task and assignment[e["group"]]==s for e in checked) for s in ("train","validation","test")}
                total = sum(counts.values())
                score += sum(abs(counts[s]/total-fraction) for s,fraction in (("train",.8),("validation",.1),("test",.1)))
            if best is None or score<best[0]:
                best = (score,assignment)
        if best is not None:
            break
    if best is None:
        raise ValueError("Unable to form task-complete disjoint splits; collect more independent groups")
    return {s:[e for e in checked if best[1][e["group"]]==s] for s in ("train","validation","test")}


def resample_indices(rows, fps):
    """Causal zero-order sample selection; no future observation/action leakage."""
    times = np.array([r["host_monotonic_ns"] for r in rows], dtype=np.int64)
    relative = (times-times[0])/1e9
    if fps<=0 or not np.isfinite(fps) or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError("Invalid source timing or output frequency")
    grid = np.arange(int(np.floor(relative[-1]*fps))+1)/fps
    indices = np.searchsorted(relative,grid,side="right")-1
    if np.any(grid-relative[indices] > 1.5/fps):
        raise ValueError("Recording gap exceeds 1.5 target periods; segment/recollect instead of filling")
    return indices, grid-relative[indices]


def convert_captures(cfg, manifest, repo_base="local/piper_chemical", train_only=False):
    from PIL import Image
    plan = plan_dataset(manifest, cfg["training"]["seed"], train_only=train_only)
    fps = cfg["robot"]["control_hz"]
    if int(fps)!=fps:
        raise ValueError("LeRobot export requires integer fps")
    # Check all timing before creating even the first split.
    selected = {}
    for entries in plan.values():
        for entry in entries:
            rows = [json.loads(line) for line in (Path(entry["path"])/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
            indices, ages = resample_indices(rows, fps)
            entry["exported_frames"] = len(indices)
            entry["max_resampling_age_s"] = float(ages.max())
            selected[entry["path"]] = (rows, indices)
    from .training import attach
    attach(cfg)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
    receipt_path = Path(cfg["paths"]["dataset_receipt"])
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    targets = {s:HF_LEROBOT_HOME/f"{repo_base}_{s}" for s in plan}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError("A target dataset exists; choose a new repo-id and receipt path")
    features = {key:{"dtype":"image","shape":(224,224,3),"names":["height","width","channel"]} for key in ("image","wrist_image")}
    features.update({k:{"dtype":"float32","shape":(7,),"names":ORDER} for k in ("state","actions")})
    receipt = {"schema":SCHEMA,"control_hz":fps,"splits":{},"source_manifest":str(Path(manifest).resolve()),"evaluation_scope":"train_only_no_held_out_evaluation" if train_only else "group_disjoint_splits"}
    for split, entries in plan.items():
        rid = f"{repo_base}_{split}"
        dataset = LeRobotDataset.create(repo_id=rid,robot_type="piper_x",fps=int(fps),features=features,use_videos=False,image_writer_threads=2,image_writer_processes=0)
        for entry in entries:
            path = Path(entry["path"])
            rows, indices = selected[str(path)]
            for index in indices:
                row = rows[index]
                frame = {"state":np.asarray(row["observation.state"],dtype=np.float32),"actions":np.asarray(row["action"],dtype=np.float32),"task":entry["prompt"]}
                for key,cam in (("image","front"),("wrist_image","wrist")):
                    with Image.open(path/row["images"][cam]["path"]) as image:
                        # Same resize-with-padding convention used by openpi live inputs.
                        from openpi_client.image_tools import resize_with_pad
                        frame[key] = resize_with_pad(np.asarray(image.convert("RGB")),224,224)
                dataset.add_frame(frame)
            dataset.save_episode()
        info = targets[split]/"meta/info.json"
        receipt["splits"][split] = {"repo_id":rid,"root":str(targets[split].resolve()),"episodes":entries,
            "info_sha256":hashlib.sha256(info.read_bytes()).hexdigest()}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding="utf-8")
    return receipt_path


def validate_receipt(cfg):
    path = Path(cfg["paths"]["dataset_receipt"])
    if not path.is_file():
        raise FileNotFoundError(f"No verified dataset receipt: {path}. Run convert-captures on reviewed recordings first.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema")!=SCHEMA or receipt.get("control_hz")!=cfg["robot"]["control_hz"]:
        raise ValueError("Dataset and robot contracts differ")
    train = receipt["splits"]["train"]
    if train["repo_id"]!=cfg["training"]["repo_id"]:
        raise ValueError("Configured training dataset does not match audited export")
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    for split in receipt["splits"].values():
        actual = (HF_LEROBOT_HOME/split["repo_id"]).resolve()
        if actual != Path(split["root"]).resolve():
            raise ValueError("Dataset moved; regenerate and validate the receipt on this machine")
        if hashlib.sha256((actual/"meta/info.json").read_bytes()).hexdigest()!=split["info_sha256"]:
            raise ValueError("Dataset metadata changed since export")
    return receipt
