import json
from pathlib import Path
import numpy as np
from .env import TitrationEnv
from .expert import ScriptedExpert
from .episodes import Recorder, read


def run(cfg, destination, target=3, seed=42, randomized=False, record=False, video=False, policy_uri=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    env = TitrationEnv(cfg)
    writer = None
    recorder = None
    policy = None
    try:
        env.reset(target, seed, randomized)
        if record:
            recorder = Recorder(destination / "episode", {"fps": cfg["simulation"]["control_hz"], "seed": seed, "randomized": randomized, "target": target, "config": cfg, "prompt": f"Dispense {target} drops into the fixed test tube using both arms."})
        if video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(destination / "rollout.mp4", fps=cfg["simulation"]["control_hz"])
        if policy_uri:
            from .policy import RemotePolicy
            try:
                policy = RemotePolicy(policy_uri)
                if policy.metadata.get("control_hz") != cfg["simulation"]["control_hz"]:
                    raise ValueError("Policy and simulation control rates differ")
            except Exception as exc:
                env.stop("policy_connection_error")
                (destination / "policy_error.txt").write_text(repr(exc), encoding="utf-8")
        expert = ScriptedExpert(env)
        while not env.done:
            obs = env.observe() if (record or policy) else None
            if writer:
                writer.append_data(obs["image"] if obs else env.render())
            if env.reason:
                action = env.command.copy()
            elif policy:
                try:
                    action = policy.action(obs)
                except Exception as exc:
                    env.stop("policy_error")
                    (destination / "policy_error.txt").write_text(repr(exc), encoding="utf-8")
                    action = env.command.copy()
            else:
                action = expert.action()
            timestamp = env.data.time
            env.step(action)
            if recorder:
                recorder.add(obs, env.last_action.copy(), timestamp)
        report = env.report()
        if recorder:
            recorder.finish(report)
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        if writer:
            writer.close()
        if policy:
            policy.close()
        env.close()


def evaluate(cfg, destination, episodes=100, randomized=False, policy_uri=None, record=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for i in range(episodes):
        report = run(cfg, root / f"run_{i:04d}", target=(3, 5, 10)[i % 3], seed=cfg["simulation"]["seed"]+i, randomized=randomized, record=record, policy_uri=policy_uri)
        reports.append(report)
        print(json.dumps({"episode": i, **report}), flush=True)
    result = {"episodes": episodes, "randomized": randomized, "controller": "pi05" if policy_uri else "scripted", "success_rate": sum(r["success"] for r in reports)/episodes, "overdrop_rate": sum(r["received"] > r["target"] for r in reports)/episodes, "mean_time": sum(r["time"] for r in reports)/episodes, "reports": reports}
    result["acceptance_passed"] = episodes >= 100 and not randomized and result["success_rate"] >= .9
    (root / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def replay(path):
    meta, rows = read(path)
    if "config" not in meta:
        raise ValueError("Replay requires simulation configuration; real episodes can only be converted")
    cfg = meta["config"]
    # Asset checkout may differ on another machine; resolve through current config.
    from .config import load
    cfg["paths"] = load()["paths"]
    env = TitrationEnv(cfg)
    maximum = 0.0
    try:
        env.reset(meta["target"], meta["seed"], meta["randomized"])
        # Recorded targets have already passed the command delay/limiter.
        env.delay.clear()
        for row in rows:
            maximum = max(maximum, float(np.max(abs(row["state"]-env.state()))))
            env.step(row["action"])
        if maximum > 1e-4:
            raise ValueError(f"Replay diverged: maximum state error {maximum}")
        return {"max_state_error": maximum, **env.report()}
    finally:
        env.close()
