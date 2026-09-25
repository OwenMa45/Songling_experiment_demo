"""Explicit project entry points into a pinned external openpi checkout."""
import dataclasses
import importlib.util
import copy
import json
from pathlib import Path
import sys
from .diagnostics import command, PINS


def attach(cfg):
    root = Path(cfg["paths"]["openpi_root"])
    revision = command(["git", "-C", str(root), "rev-parse", "HEAD"])
    if not revision["ok"] or revision.get("stdout") != PINS["openpi"]:
        raise RuntimeError(f"openpi checkout must match {PINS['openpi']}; use PIPER_OPENPI_ROOT to select the pinned submodule")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "packages/openpi-client/src"))
    return root


def configuration(cfg, steps=None):
    if steps is not None and steps < 1:
        raise ValueError("Training steps must be positive")
    attach(cfg)
    from openpi import transforms as t
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as c, weight_loaders
    single = "robot" in cfg
    if single:
        from .single_arm import Inputs, Outputs
    else:
        from .transforms import Inputs, Outputs

    @dataclasses.dataclass(frozen=True)
    class PiperData(c.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            delta_mask = t.make_bool_mask(6, -1) if single else t.make_bool_mask(6, -1, 6, -1)
            mapping = t.Group(inputs=[Inputs()], outputs=[Outputs()]).push(inputs=[t.DeltaActions(delta_mask)], outputs=[t.AbsoluteActions(delta_mask)])
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=t.Group(inputs=[t.RepackTransform({"state": "state", "image": "image", "wrist_image": "wrist_image", "actions": "actions", "prompt": "prompt"})]), data_transforms=mapping, model_transforms=c.ModelTransformFactory()(model_config), prompt_from_task=True)

    training = cfg["training"]
    model = Pi0Config(pi05=True, action_horizon=training["action_horizon"], paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    output = Path(cfg["paths"]["output"])
    metadata = {"action_schema":cfg["robot"]["schema"], "action_dim":7, "control_hz":cfg["robot"]["control_hz"]} if single else {"action_schema":"piper-titration-v1","action_dim":14,"control_hz":cfg["simulation"]["control_hz"]}
    return c.TrainConfig(name="pi05_piper_chemical_lora" if single else "pi05_piper_lora", exp_name=training["experiment"], model=model, freeze_filter=model.get_freeze_filter(), ema_decay=None, data=PiperData(repo_id=training["repo_id"]), weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(cfg["paths"]["checkpoint"])/"params")), batch_size=training["batch_size"], num_workers=0, num_train_steps=steps or training["steps"], save_interval=1000, log_interval=1 if steps else 100, wandb_enabled=False, assets_base_dir=str(output/"training_assets"), checkpoint_base_dir=str(output/"checkpoints"), seed=training["seed"], policy_metadata=metadata)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def execute(cfg, mode, steps=None, checkpoint=None, port=8000):
    if mode == "serve" and checkpoint:
        saved = Path(checkpoint).resolve().parent / "piper_run.json"
        if not saved.is_file():
            raise FileNotFoundError(f"Missing training configuration: {saved}")
        original = json.loads(saved.read_text(encoding="utf-8"))
        cfg = copy.deepcopy(cfg)
        cfg["training"] = original["training"]
        if "robot" in original:
            cfg["robot"] = original["robot"]
        else:
            cfg.pop("robot",None)
            cfg["simulation"] = original["simulation"]
    root = attach(cfg)
    if "robot" in cfg and mode in ("norms", "train"):
        from .capture_dataset import validate_receipt
        validate_receipt(cfg)
    config = configuration(cfg, steps)
    if mode == "norms":
        # Use the official computation, registering only within this process.
        from openpi.training import config as registry
        registry._CONFIGS_DICT[config.name] = config
        module(root/"scripts/compute_norm_stats.py", "piper_openpi_norms").main(config.name)
    elif mode == "train":
        params = Path(cfg["paths"]["checkpoint"])/"params"
        if not params.is_dir() or not any(params.iterdir()):
            raise FileNotFoundError(f"Download the complete JAX pi05_base checkpoint first: {params}")
        if config.checkpoint_dir.exists():
            raise FileExistsError(f"Choose a new training.experiment: {config.checkpoint_dir}")
        try:
            module(root/"scripts/train.py", "piper_openpi_train").main(config)
        finally:
            # Also retain the config if a run fails after producing an intermediate checkpoint.
            if config.checkpoint_dir.is_dir():
                snapshot = copy.deepcopy(cfg)
                snapshot["training"]["steps"] = config.num_train_steps
                (config.checkpoint_dir/"piper_run.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    elif mode == "serve":
        if not checkpoint or not Path(checkpoint).is_dir():
            raise ValueError("--checkpoint must point to a trained step directory containing params and assets")
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        policy = create_trained_policy(config, checkpoint)
        WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=port, metadata=config.policy_metadata).serve_forever()
    else:
        raise ValueError(mode)
