"""Audit capture folders without changing recordings or synthesizing actions."""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p95": ordered[round((len(ordered)-1)*.95)], "max": ordered[-1]}


def vector(value):
    return (isinstance(value, list) and len(value) == 7
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in value))


def prepared_checks(root, meta, rows):
    """Validate v2 evidence and provenance, without certifying hardware ourselves."""
    failures = []
    try:
        records = []
        for field, path_key, hash_key in (
            ("physical_feedback_source_verification", "record", "sha256"),
            ("action_source_verification", "hold_behavior_record", "hold_behavior_sha256"),
        ):
            ref = meta[field]
            path = (root/ref[path_key]).resolve()
            if not path.is_relative_to(root.parent.parent):
                raise ValueError("Verification path escapes prepared package")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != ref[hash_key]:
                raise ValueError("Verification hash mismatch")
            records.append(json.loads(data))
        identity, hold = records
        if identity.get("result") != "pass" or identity.get("verified") is not True:
            raise ValueError("Identity verification did not pass")
        for record in (identity, hold["verified_configuration"]):
            for key in ("firmware_profile", "can_interface", "topology"):
                if record.get(key) != meta.get(key):
                    raise ValueError("Verification configuration mismatch")
        if hold.get("result") != "pass" or hold["conclusion"].get("sample_and_hold_empirically_verified") is not True:
            raise ValueError("Hold verification did not pass")
        policy = hold["dataset_policy"]
        empirical = hold["conclusion"]["max_empirically_observed_hold_gap_ms"]
        caps = {"recent_recorded_target":200.0,
                "verified_internal_sample_and_hold":policy["max_internal_hold_ms"],
                "verified_terminal_sample_and_hold":policy["max_terminal_hold_ms"]}
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0 < v <= empirical for v in caps.values()):
            raise ValueError("Invalid hold thresholds")
        if meta.get("action_source_verified") is not True or meta["action_source_verification"].get("no_future_state_as_action") is not True:
            raise ValueError("Action source not reviewed")
        review = meta.get("human_review",{})
        if review.get("reviewed") is not True or review.get("visual_ok") is not True:
            raise ValueError("Prepared episode not reviewed")
        terminal = False
        previous_source = None
        for i,row in enumerate(rows):
            source = row["source_sample_index"]
            if row["sample_index"] != i or (previous_source is not None and source != previous_source+1):
                raise ValueError("Prepared episode contains discontinuous indices")
            previous_source = source
            p = row["action_provenance"]
            approval = p["approval"]
            age, complete, skew = (p[k] for k in ("command_oldest_part_age_ms","command_complete_age_ms","target_part_skew_ms"))
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (age,complete,skew)):
                raise ValueError("Invalid action timing")
            if approval not in caps or not 0 <= complete <= age <= caps[approval] or not 0 <= skew <= 5 or abs(age-complete-skew)>1e-3:
                raise ValueError("Action timing exceeds verified bounds")
            if terminal and approval != "verified_terminal_sample_and_hold":
                raise ValueError("Terminal hold is not trailing")
            terminal |= approval == "verified_terminal_sample_and_hold"
            if p.get("hold_verification_record_sha256") != meta["action_source_verification"]["hold_behavior_sha256"] or p.get("no_future_state_inference") is not True:
                raise ValueError("Action provenance mismatch")
            lines = p["source_can_lines"]
            if set(lines) != {"155","156","157","159"} or not all(isinstance(n,int) and n>0 for n in lines.values()):
                raise ValueError("Missing CAN provenance")
        manifest = [json.loads(line) for line in (root/"images_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = {(i,c):r["images"][c]["path"] for i,r in enumerate(rows) for c in ("front","wrist")}
        seen = set()
        for entry in manifest:
            key = (entry["sample_index"],entry["camera"])
            if key in seen or expected.get(key) != entry["prepared_path"]:
                raise ValueError("Image manifest indexing mismatch")
            seen.add(key)
            path = (root/entry["prepared_path"]).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Image manifest hash mismatch")
        if seen != set(expected):
            raise ValueError("Image manifest incomplete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append("prepared_evidence_invalid: " + str(exc))
    return failures


def audit(root, decode_images=False):
    root = Path(root).resolve()
    meta_path, samples_path = root/"metadata.json", root/"samples.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows, parse_errors = [], []
    for line_no, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            parse_errors.append({"line": line_no, "error": str(exc)})
    blockers, warnings = [], []
    invalid_states = [i for i, r in enumerate(rows) if not vector(r.get("observation.state"))]
    invalid_actions = [i for i, r in enumerate(rows) if not vector(r.get("action"))]
    if parse_errors:
        blockers.append("invalid_jsonl")
    if not rows:
        blockers.append("empty_episode")
    if invalid_states:
        blockers.append("invalid_states")
    if invalid_actions:
        blockers.append("missing_or_invalid_command_actions")
    if meta.get("samples") != len(rows):
        blockers.append("metadata_frame_count_mismatch")
    if meta.get("physical_feedback_source_verified") is not True:
        blockers.append("physical_feedback_source_not_verified")
    if meta.get("ready_for_policy_training") is False:
        blockers.append("recording_explicitly_not_training_ready")
    if meta.get("task_success") is not True:
        blockers.append("successful_demonstration_not_reviewed")
    if meta.get("state_units") != ["rad"]*6+["m"]:
        blockers.append("state_units_not_supported")
    if meta.get("action_units") != ["rad"]*6+["m"]:
        blockers.append("command_units_not_verified")
    if meta.get("format") == "piper_x_single_arm_prepared_v2":
        if meta.get("action_semantics") != "absolute recorded CAN target; verified sample-and-hold where target transmission pauses":
            blockers.append("command_semantics_not_verified")
        blockers.extend(prepared_checks(root,meta,rows))
        warnings.append("prepared_verification_records_are_operator_attestations_not_independent_hardware_tests")
        if "assume" in meta.get("human_review",{}).get("visual_source",""):
            warnings.append("visual_quality_assumed_by_preprocessor_review_before_robot_deployment")
    elif not str(meta.get("action_semantics", "")).startswith("absolute_leader_transmitted_joint_targets"):
        blockers.append("command_semantics_not_verified")
    if meta.get("error"):
        blockers.append("recorder_error")
    if "interrupted" in meta.get("recording_status", ""):
        warnings.append("recording_interrupted_manual_completion_review_required")
    timestamps = [r.get("host_monotonic_ns") for r in rows]
    valid_times = all(isinstance(t, int) and not isinstance(t, bool) for t in timestamps)
    differences = [(b-a)/1e9 for a,b in zip(timestamps, timestamps[1:])] if valid_times else []
    if not valid_times or any(d <= 0 for d in differences):
        blockers.append("invalid_or_nonmonotonic_timestamps")
    duration = (timestamps[-1]-timestamps[0])/1e9 if valid_times and len(rows)>1 else 0
    nominal = meta.get("requested_fps")
    gaps = sum(d > 1.5/nominal for d in differences) if isinstance(nominal, (int,float)) and nominal>0 else None
    if gaps:
        warnings.append("sampling_gaps_require_segment_review")
    missing, unreadable, cameras, stale = [], [], {}, Counter()
    for camera in meta.get("cameras", {}):
        sequences, offsets, hashes, brightness = [], [], [], []
        for i, r in enumerate(rows):
            info = r.get("images", {}).get(camera)
            if not isinstance(info, dict) or not isinstance(info.get("path"), str):
                missing.append({"sample": i, "camera": camera, "path": None})
                continue
            p = (root/info["path"]).resolve()
            if not p.is_relative_to(root):
                unreadable.append({"sample": i, "camera": camera, "error": "path escapes episode"})
                continue
            sequences.append(info.get("sequence"))
            offset = info.get("relative_to_sample_ms")
            if isinstance(offset, (int,float)) and math.isfinite(offset):
                offsets.append(offset)
            if not p.is_file():
                missing.append({"sample": i, "camera": camera, "path": info["path"]})
                continue
            hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
            if decode_images:
                from PIL import Image, ImageStat
                try:
                    with Image.open(p) as im:
                        im.load()
                        if [im.height, im.width, 3] != info.get("shape_hwc"):
                            raise ValueError("image dimensions differ from recorded shape")
                        brightness.append({"sample": i, "mean_luma": ImageStat.Stat(im.convert("L")).mean[0]})
                except Exception as exc:
                    unreadable.append({"sample": i, "camera": camera, "error": str(exc)})
        cameras[camera] = {"frames": len(sequences), "unique_sequences": len(set(sequences)),
                           "reused_sequence_count": len(sequences)-len(set(sequences)),
                           "unique_file_hashes": len(set(hashes)), "offset_ms": summary(offsets),
                           "mean_luma": summary([b["mean_luma"] for b in brightness]),
                           "first_15_luma": brightness[:15]}
        if len(set(sequences)) < len(sequences):
            warnings.append(f"{camera}_reused_camera_frames")
    if not cameras:
        blockers.append("no_camera_observations")
    if missing:
        blockers.append("missing_images")
    if unreadable:
        blockers.append("invalid_images")
    if not decode_images:
        blockers.append("image_decode_not_verified")
    for r in rows:
        for name, stream in r.get("streams", {}).items():
            age = stream.get("host_since_timestamp_change_s")
            if isinstance(age,(float,int)) and age > meta.get("max_cache_age_s", .2):
                stale[name] += 1
    if stale:
        blockers.append("stale_robot_feedback")
    state_ranges = []
    if rows and not invalid_states:
        for j in range(7):
            v = [r["observation.state"][j] for r in rows]
            state_ranges.append({"coordinate":j, "min":min(v),"max":max(v),"unique_values":len(set(v))})
    can_ids = Counter()
    candidate_times = {key: [] for key in ("155", "156", "157", "159")}
    if (root/"can_raw.log").is_file():
        for line in (root/"can_raw.log").open(encoding="utf-8"):
            match = re.search(r"\s([0-9a-fA-F]+)#[0-9a-fA-F]*", line)
            if match:
                can_id = match[1].upper()
                can_ids[can_id] += 1
                stamp = re.match(r"\(([0-9.]+)\)", line)
                if can_id in candidate_times and stamp:
                    candidate_times[can_id].append(float(stamp[1]))
    candidate_ages = []
    candidate_complete = 0
    for times in candidate_times.values():
        times.sort()
    for row in rows:
        if not isinstance(row.get("host_time_ns"), int):
            continue
        wall = row["host_time_ns"] / 1e9
        preceding = []
        for times in candidate_times.values():
            i = bisect.bisect_right(times, wall)-1
            if i >= 0:
                preceding.append(times[i])
        if len(preceding) == 4:
            age = (wall-min(preceding))*1000
            candidate_ages.append(age)
            candidate_complete += int(age <= 1000*meta.get("max_cache_age_s", .2))
    report = {"episode":str(root),"task":meta.get("task"), "frames":len(rows),
              "duration_s":duration,"effective_fps":(len(rows)-1)/duration if duration>0 else None,
              "frame_interval_s":summary(differences),"gaps_over_1_5_periods":gaps,
              "null_action_count":sum(r.get("action") is None for r in rows),
              "invalid_action_count":len(invalid_actions),"invalid_state_count":len(invalid_states),
              "state_ranges":state_ranges,"cameras":cameras,"missing_images":missing,
              "invalid_images":unreadable,"stale_stream_counts":dict(stale),
              "can_id_counts":dict(sorted(can_ids.items())),"parse_errors":parse_errors,
              "candidate_can_timing": {"ids":list(candidate_times), "samples_with_all_four_recent_preceding_frames":candidate_complete,
                                       "oldest_frame_age_ms":summary(candidate_ages),
                                       "interpretation":"IDs are candidates from official Piper-family protocol; no payload decoding or physical sender verification; not action labels"},
              "metadata_sha256":hashlib.sha256(meta_path.read_bytes()).hexdigest(),
              "samples_sha256":hashlib.sha256(samples_path.read_bytes()).hexdigest(),
              "blockers":blockers,"warnings":warnings,"training_ready":not blockers,
              "review_scope":"structural audit; image content and physical source need separate human verification"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.path] if (args.path/"metadata.json").is_file() else sorted(p.parent for p in args.path.rglob("metadata.json"))
    reports = [audit(p,args.decode_images) for p in roots]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"episodes":reports,"ready":sum(r["training_ready"] for r in reports),"total":len(reports)},ensure_ascii=False,indent=2),encoding="utf-8")
    for r in reports:
        print(json.dumps({k:r[k] for k in ("episode","frames","effective_fps","training_ready","blockers")},ensure_ascii=False))
    return 0 if reports and all(r["training_ready"] for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
