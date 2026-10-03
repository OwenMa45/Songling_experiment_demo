"""Synchronous, bounded checkpoint writes for the 64 GiB training host."""
from pathlib import Path


def initialize_checkpoint_dir(checkpoint_dir, *, keep_period, overwrite, resume):
    import orbax.checkpoint as ocp
    from openpi.training.checkpoints import CallbackHandler

    # Project training always uses a fresh experiment; never remove old work.
    if overwrite or resume:
        raise ValueError("Use a fresh experiment for bounded checkpoint training")
    path = Path(checkpoint_dir).resolve()
    path.mkdir(parents=True, exist_ok=False)
    manager = ocp.CheckpointManager(
        path,
        item_handlers={
            "assets": CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(save_concurrent_gb=2),
            "train_state": ocp.PyTreeCheckpointHandler(save_concurrent_gb=2),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, keep_period=keep_period, create=False,
            enable_async_checkpointing=False,
        ),
    )
    return manager, False
