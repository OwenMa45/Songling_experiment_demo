"""Read-only scan of all hold ages; does not approve data or alter thresholds."""
import argparse
import json
import math
from pathlib import Path


def summarize(manifest):
    manifest = Path(manifest).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))["episodes"]
    result = []
    for entry in entries:
        root = (manifest.parent/entry["path"]).resolve()
        record = {"episode":root.name,"holds":{}}
        try:
            meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
            ref = meta["action_source_verification"]["hold_behavior_record"]
            verification = json.loads((root/ref).read_text(encoding="utf-8"))
            empirical = verification["conclusion"]["max_empirically_observed_hold_gap_ms"]
            record["declared_empirical_hold_ms"] = empirical
            policy = verification["dataset_policy"]
            for line in (root/"samples.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                p = row["action_provenance"]
                label = p["approval"]
                if label not in ("verified_internal_sample_and_hold","verified_terminal_sample_and_hold"):
                    continue
                age = p["command_oldest_part_age_ms"]
                if not isinstance(age,(int,float)) or not math.isfinite(age) or age < 0:
                    raise ValueError("Invalid hold age")
                cap = policy["max_internal_hold_ms" if label=="verified_internal_sample_and_hold" else "max_terminal_hold_ms"]
                stat = record["holds"].setdefault(label,{"count":0,"max_age_ms":-1,"over_policy_count":0,"over_declared_empirical_count":0,"policy_cap_ms":cap})
                stat["count"] += 1
                stat["over_policy_count"] += int(age>cap)
                stat["over_declared_empirical_count"] += int(age>empirical)
                if age > stat["max_age_ms"]:
                    stat.update(max_age_ms=age,sample_index=row["sample_index"],source_sample_index=row.get("source_sample_index"))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            record["error"] = str(exc)
        result.append(record)
    return {"scope":"Statistics only; empirical values are declarations, not revalidated evidence or training approval", "episodes":result}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    args=p.parse_args()
    report=summarize(args.manifest)
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    raise SystemExit(int(any("error" in r for r in report["episodes"])))
