"""Bind normalization to actual data/coordinates and test the official inverse path.

No robot connection. A software round trip does not calibrate hardware zero points.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from .single_arm import ORDER, coordinate_contract
from .diagnostics import PINS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(root):
    """Hash numeric supervision + metadata. Images are covered by capture audit,
    not by this fingerprint; do not claim a full image-content dataset digest.
    """
    root = Path(root)
    paths = sorted({*root.glob("data/**/*.parquet"), *root.glob("meta/**/*.json"),
                    *root.glob("meta/**/*.jsonl")})
    if not any(p.suffix==".parquet" for p in paths):
        raise ValueError(f"No LeRobot parquet supervision under {root}")
    manifest = {p.relative_to(root).as_posix():sha256(p) for p in paths}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def norm_path(config):
    return Path(config.assets_dirs)/config.data.repo_id/"norm_stats.json"


def stamp_path(config):
    return norm_path(config).with_name("piper_norm_audit.json")


def binding(cfg, config, receipt):
    return {
        "contract":coordinate_contract(cfg["robot"],cfg["training"]["action_horizon"]),
        "repo_id":config.data.repo_id, "openpi_commit":PINS["openpi"],
        "dataset_receipt_sha256":sha256(cfg["paths"]["dataset_receipt"]),
        "train_numeric_metadata_sha256":dataset_fingerprint(receipt["splits"]["train"]["root"]),
    }


def inspect_stats(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["norm_stats"]
    if set(raw) != {"state","actions"}:
        raise ValueError("Norms must contain exactly state and actions")
    summary = {}
    for name in ("state","actions"):
        arrays = {key:np.asarray(raw[name].get(key),dtype=float) for key in ("mean","std","q01","q99")}
        if any(a.shape!=(7,) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f"{name} norms must have seven finite values per statistic")
        if np.any(arrays["std"]<0) or np.any(arrays["q99"]<arrays["q01"]):
            raise ValueError(f"Invalid {name} norm variance/quantile order")
        span = arrays["q99"]-arrays["q01"]
        summary[name] = {"mean":arrays["mean"].tolist(),"std":arrays["std"].tolist(),
                         "q01":arrays["q01"].tolist(),"q99":arrays["q99"].tolist(),
                         "near_constant_coordinates":[ORDER[i] for i in np.flatnonzero(span<1e-5)]}
    return summary


def roundtrip_arrays(state, actions, stats, transforms):
    """Use upstream Normalize/Unnormalize and the production output transforms."""
    from openpi import transforms as t
    from .single_arm import DELTA_MASK
    state,actions = np.asarray(state,dtype=np.float32),np.asarray(actions,dtype=np.float32)
    if state.shape!=(7,) or actions.ndim!=2 or actions.shape[1]!=7:
        raise ValueError("Round-trip expects state[7], actions[H,7]")
    expected = actions.copy()
    expected[:,:6] -= state[:6]
    delta = t.DeltaActions(DELTA_MASK)({"state":state.copy(),"actions":actions.copy()})
    np.testing.assert_allclose(delta["actions"],expected,rtol=0,atol=1e-6)
    normalized = t.Normalize(stats,use_quantiles=True)(delta)
    padded = t.PadStatesAndActions(32)(normalized)
    if not np.isfinite(padded["state"]).all() or not np.isfinite(padded["actions"]).all():
        raise ValueError("Nonfinite normalized values")
    restored = t.Unnormalize(stats,use_quantiles=True)(copy.deepcopy(padded))
    np.testing.assert_allclose(restored["state"][:7],state,rtol=0,atol=2e-6)
    result = t.compose(transforms.outputs)(restored)["actions"]
    np.testing.assert_allclose(result,actions,rtol=0,atol=2e-6)
    return result, padded


def runtime_roundtrip(config, sample_count=64):
    """Exercise real LeRobot horizon loading, actual mapping and official inverse.
    Samples span the dataset; this is not robot task evaluation.
    """
    from openpi import transforms as t
    from openpi.training import data_loader
    from openpi.shared import normalize
    data = config.data.create(config.assets_dirs,config.model)
    if not data.use_quantile_norm:
        raise ValueError("Single-arm pi05 must use quantile normalization")
    stats = normalize.load(norm_path(config).parent)
    dataset = data_loader.create_torch_dataset(data,config.model.action_horizon,config.model)
    if len(dataset)<1:
        raise ValueError("Empty training dataset")
    indices = np.unique(np.linspace(0,len(dataset)-1,min(sample_count,len(dataset)),dtype=int))
    errors,outside,maximum = [], {k:np.zeros(7,dtype=int) for k in stats}, {k:0.0 for k in stats}
    totals = {k:0 for k in stats}
    for i in indices:
        raw = t.compose(data.repack_transforms.inputs)(dataset[int(i)])
        state,actions = np.asarray(raw["state"]),np.asarray(raw["actions"])
        if actions.shape!=(config.model.action_horizon,7):
            raise ValueError("LeRobot action chunk shape differs from configured horizon")
        transformed = t.compose(data.data_transforms.inputs)(copy.deepcopy(raw))
        expected = actions.copy(); expected[:,:6] -= state[:6]
        np.testing.assert_allclose(transformed["actions"],expected,rtol=0,atol=1e-6)
        np.testing.assert_allclose(transformed["state"],state,rtol=0,atol=1e-6)
        result,padded = roundtrip_arrays(state,actions,stats,data.data_transforms)
        errors.append(float(np.max(np.abs(result-actions))))
        for key in stats:
            array = padded[key][...,:7].reshape(-1,7)
            outside[key] += np.sum(np.abs(array)>1,axis=0)
            maximum[key] = max(maximum[key],float(np.max(np.abs(array))))
            totals[key] += len(array)
    return {"sampled_chunk_count":len(indices),"total_dataset_frames":len(dataset),
            "max_roundtrip_error_rad_or_m":max(errors),
            "sample_fraction_outside_minus1_plus1":{k:(outside[k]/totals[k]).tolist() for k in stats},
            "sample_max_abs_normalized":maximum,
            "note":"Quantile tails outside [-1,1] are not automatically clipped or errors; investigate large values and near-constant axes."}


def finish_norms(cfg,config,receipt,before):
    after = binding(cfg,config,receipt)
    if before != after:
        raise ValueError("Dataset/config changed during norm computation; recompute norms")
    report = {"binding":after,"norm_sha256":sha256(norm_path(config)),
              "stats":inspect_stats(norm_path(config)),"roundtrip":runtime_roundtrip(config),
              "hardware_calibration_verified_by_this_check":False}
    path = stamp_path(config)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


def check_norms(cfg,config,receipt,roundtrip=False):
    path = stamp_path(config)
    if not path.is_file():
        raise ValueError(f"No norm audit at {path}. Run norms once with this version; no dataset reconversion required.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["binding"] != binding(cfg,config,receipt):
        raise ValueError("Dataset, coordinate convention or horizon changed since norms; recompute norms")
    if report["norm_sha256"] != sha256(norm_path(config)):
        raise ValueError("Norm file changed after audit; recompute norms")
    inspect_stats(norm_path(config))
    if roundtrip:
        report["roundtrip"] = runtime_roundtrip(config)
    return report


def check_checkpoint(cfg,config,checkpoint,report):
    if not report:
        raise ValueError("Checkpoint lacks normalization audit; do not silently pair old weights with new norms")
    if report["binding"]["contract"] != coordinate_contract(cfg["robot"],cfg["training"]["action_horizon"]):
        raise ValueError("Checkpoint coordinate contract differs from current runtime")
    path = Path(checkpoint)/"assets"/config.data.repo_id/"norm_stats.json"
    # Upstream load/save may reformat JSON: compare numerical statistics, not text.
    if inspect_stats(path) != report["stats"]:
        raise ValueError("Checkpoint norms differ from audited training norms")


def coordinate_episode(path, robot, horizon):
    """NumPy-only read-only diagnostics; NEVER writes trainable norm statistics."""
    from .capture_dataset import resample_indices
    root = Path(path)
    meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (root/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if meta.get("state_order") != ORDER or meta.get("state_units") != ["rad"]*6+["m"] or meta.get("action_units") != ["rad"]*6+["m"]:
        raise ValueError("Episode order/units differ from single-arm contract")
    states = np.asarray([r["observation.state"] for r in rows],dtype=np.float32)
    actions = np.asarray([r["action"] for r in rows],dtype=np.float32)
    if states.shape != (len(rows),7) or actions.shape != states.shape or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Invalid state/action array")
    indices,ages = resample_indices(rows,robot["control_hz"])
    s,a = states[indices],actions[indices]
    chunks = a[np.minimum(np.arange(len(a))[:,None]+np.arange(horizon),len(a)-1)]
    deltas = chunks.copy(); deltas[:,:,:6] -= s[:,None,:6]
    def describe(values):
        flat=values.reshape(-1,7)
        q01,q99=np.quantile(flat,[.01,.99],axis=0)
        return {"min":flat.min(axis=0).tolist(),"max":flat.max(axis=0).tolist(),
                "q01_diagnostic":q01.tolist(),"q99_diagnostic":q99.tolist(),
                "near_constant":[ORDER[i] for i in np.flatnonzero(q99-q01<1e-5)]}
    return {"episode":str(root.resolve()),"coordinate_contract":coordinate_contract(robot,horizon),
            "raw_frames":len(rows),"exported_frames":len(indices),
            "resampling_duplicate_selections":len(indices)-len(np.unique(indices)),
            "max_resampling_age_ms":float(ages.max()*1000),
            "state":describe(s),"absolute_targets":describe(a),"model_target_before_norm":describe(deltas),
            "negative_gripper_target_count":int(np.sum(a[:,6]<0)),
            "max_absolute_target_step":np.abs(np.diff(a,axis=0)).max(axis=0).tolist(),
            "source_sha256":sha256(root/"samples.jsonl"),
            "scope":"single episode numerical diagnostics; numpy exact quantiles, not official histogram norms. No hardware calibration, exposure synchronization or model performance claim."}
