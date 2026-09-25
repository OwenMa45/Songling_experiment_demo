"""Offline diagnostic candidates only. Never edits samples or certifies hardware identity."""
import argparse
import bisect
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SDK_COMMIT = "841a625f5f4920e776f20b934eb13048b747e6d0"
TARGET_IDS = (0x155, 0x156, 0x157, 0x159)
FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A8)
PATTERN = re.compile(r"^\((\d+\.\d+)\)\s+(\S+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)\s*$")


@dataclass(frozen=True)
class Frame:
    time_ns: int
    channel: str
    identifier: int
    payload: bytes
    line: int


def read_frames(path, channel="can0"):
    frames, previous = [], None
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = PATTERN.fullmatch(line)
        if not match:
            raise ValueError(f"Unsupported candump line {line_no}; no silent dropping")
        timestamp, name, identifier, payload = match.groups()
        if name != channel:
            raise ValueError(f"Unexpected CAN interface {name} at line {line_no}")
        now = int(Decimal(timestamp)*1_000_000_000)
        if previous is not None and now < previous:
            raise ValueError(f"Wall-clock rollback at line {line_no}")
        previous = now
        frame = Frame(now, name, int(identifier,16), bytes.fromhex(payload), line_no)
        if frame.identifier in (*TARGET_IDS,*FEEDBACK_IDS,0x151) and len(frame.payload)!=8:
            raise ValueError(f"Invalid payload length at line {line_no}")
        frames.append(frame)
    return frames


def decode(frame):
    d = frame.payload
    if len(d)!=8:
        raise ValueError("Expected eight bytes")
    if frame.identifier in (*TARGET_IDS[:3], *FEEDBACK_IDS[:3]):
        return [int.from_bytes(d[i:i+4],"big",signed=True)*1e-3*math.pi/180 for i in (0,4)]
    if frame.identifier in (0x159,0x2A8):
        if frame.identifier == 0x2A8 and d[7]!=0:
            raise ValueError("Non-width gripper feedback is unsupported")
        if frame.identifier == 0x159 and (d[6] not in (1,3) or d[7]!=0):
            raise ValueError("Gripper target disabled or requests zeroing")
        return [int.from_bytes(d[:4],"big",signed=True)*1e-6]
    raise ValueError("Unsupported coordinate frame")


def target_groups(frames, max_span_ns=5_000_000):
    """Require one complete ordered cycle; do not splice several cycles together."""
    pending, groups = [], []
    rejected = Counter()
    for frame in frames:
        if frame.identifier not in TARGET_IDS:
            continue
        if frame.identifier == TARGET_IDS[0]:
            if pending:
                rejected["incomplete_group"] += 1
            pending = [frame]
            continue
        if not pending or frame.identifier != TARGET_IDS[len(pending)]:
            rejected["out_of_order_or_missing_start"] += 1
            pending = []
            continue
        pending.append(frame)
        if len(pending)==4:
            if pending[-1].time_ns-pending[0].time_ns > max_span_ns:
                rejected["group_span_exceeded"] += 1
            else:
                try:
                    values = [x for f in pending for x in decode(f)]
                    groups.append({"start_ns":pending[0].time_ns,"end_ns":pending[-1].time_ns,
                                   "lines":[f.line for f in pending],"action":values})
                except ValueError:
                    rejected["invalid_gripper_mode_or_status"] += 1
            pending = []
    if pending:
        rejected["incomplete_group"] += 1
    return groups,dict(rejected)


def match_group(groups, ends, now, max_age_ns):
    index = bisect.bisect_right(ends,now)-1
    if index<0:
        return None,"no_complete_preceding_group"
    group = groups[index]
    if now-group["start_ns"]>max_age_ns:
        return None,"target_group_stale"
    return group,None


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"min":values[0],"median":statistics.median(values),"p95":values[round(.95*(len(values)-1))],"max":values[-1]}


def recover(root, destination, max_age_ms=200, max_group_span_ms=5):
    if not all(math.isfinite(v) and v>0 for v in (max_age_ms,max_group_span_ms)):
        raise ValueError("Timing thresholds must be finite and positive")
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    if destination.is_relative_to(root):
        raise ValueError("Keep all derived diagnostics outside the original episode")
    if destination.exists():
        raise FileExistsError(destination)
    meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("sdk_commit_reference")!=SDK_COMMIT:
        raise ValueError("SDK revision differs from reviewed decoder")
    rows = [json.loads(x) for x in (root/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    times = [r["host_time_ns"] for r in rows]
    if not rows or any(not isinstance(t,int) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("Invalid observation wall-clock sequence")
    offsets = [r["host_time_ns"]-r["host_monotonic_ns"] for r in rows]
    if max(offsets)-min(offsets)>50_000_000:
        raise ValueError("Wall/monotonic clock drift exceeds 50 ms")
    frames = read_frames(root/"can_raw.log",meta["can_interface"])
    observed = Counter(f.identifier for f in frames)
    offset_ids = [i+offset for offset in (0x10,0x20) for i in TARGET_IDS]
    if any(observed[i] for i in offset_ids):
        raise ValueError("Offset control IDs present; resolve topology before default-ID decoding")
    groups,rejected = target_groups(frames,round(max_group_span_ms*1e6))
    ends = [g["end_ns"] for g in groups]
    streams = {i:[f for f in frames if f.identifier==i] for i in (*FEEDBACK_IDS,0x151)}
    stream_times = {i:[f.time_ns for f in fs] for i,fs in streams.items()}
    output, errors, reasons = [], [[] for _ in range(7)], Counter()
    for row in rows:
        now = row["host_time_ns"]
        group,reason = match_group(groups,ends,now,round(max_age_ms*1e6))
        mode_index = bisect.bisect_right(stream_times[0x151],now)-1
        mode = streams[0x151][mode_index] if mode_index>=0 else None
        # Treat 151 as a mode latch, not an action freshness signal; report its timestamp.
        if group and (mode is None or mode.payload[0]!=1 or mode.payload[1]!=1 or mode.payload[3] not in (0,0xAD)):
            group,reason = None,"joint_control_mode_not_observed"
        comparison,feedback_lines,feedback_times = [],[],[]
        feedback_reason = None
        for identifier in FEEDBACK_IDS:
            index = bisect.bisect_right(stream_times[identifier],now)-1
            if index<0:
                feedback_reason = "missing_preceding_feedback"
                break
            frame = streams[identifier][index]
            if now-frame.time_ns>round(max_age_ms*1e6):
                feedback_reason = "stale_feedback"
                break
            try:
                comparison.extend(decode(frame))
            except ValueError:
                feedback_reason = "unsupported_feedback_mode"
                break
            feedback_lines.append(frame.line)
            feedback_times.append(frame.time_ns)
        difference = None
        if feedback_reason is None and len(comparison)==7:
            state = row["observation.state"]
            if not isinstance(state,list) or len(state)!=7 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in state):
                raise ValueError("Invalid original state")
            difference = [abs(a-b) for a,b in zip(comparison,state)]
            for values,value in zip(errors,difference):
                values.append(value)
        if reason:
            reasons[reason] += 1
        output.append({"sample_index":row["sample_index"],"host_time_ns":now,
                       "candidate_action":group["action"] if group else None,
                       "target_group":group,"candidate_rejection":reason,
                       "mode_frame_line":mode.line if mode else None,"mode_frame_time_ns":mode.time_ns if mode else None,
                       "feedback_decoded":comparison if feedback_reason is None else None,
                       "feedback_lines":feedback_lines,"feedback_times_ns":feedback_times,
                       "absolute_state_difference":difference,"feedback_rejection":feedback_reason,
                       "physical_feedback_source_verified":False,"ready_for_policy_training":False})
    segments = []
    for r in output:
        if r["candidate_action"] is not None:
            i=r["sample_index"]
            if segments and i==segments[-1][1]+1:
                segments[-1][1]=i
            else:
                segments.append([i,i])
    report = {"episode":str(root),"sdk_commit":SDK_COMMIT,
              "source_hashes":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ("metadata.json","samples.jsonl","can_raw.log")},
              "timing_thresholds_ms":{"max_target_age":max_age_ms,"max_group_span":max_group_span_ms},
              "complete_target_groups":len(groups),"rejected_groups":rejected,"samples":len(rows),
              "candidate_samples":sum(r["candidate_action"] is not None for r in output),
              "candidate_segments_inclusive":segments,"rejection_counts":dict(reasons),
              "feedback_absolute_errors_rad_then_metres":[stats(v) for v in errors],
              "feedback_comparison_note":"Latest preceding CAN values versus asynchronous SDK snapshot; not identity verification, latency calibration or a guaranteed atomic six-joint snapshot.",
              "wall_minus_monotonic_span_ms":(max(offsets)-min(offsets))/1e6,
              "decoder_assumption":"one observed CAN interface, default IDs; device identity not inferred",
              "physical_feedback_source_verified":False,"ready_for_policy_training":False}
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"candidates.jsonl").write_text("\n".join(json.dumps(r,allow_nan=False) for r in output)+"\n",encoding="utf-8")
    (destination/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-age-ms",type=float,default=200)
    p.add_argument("--max-group-span-ms",type=float,default=5)
    args=p.parse_args()
    print(json.dumps(recover(args.episode,args.output,args.max_age_ms,args.max_group_span_ms),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
