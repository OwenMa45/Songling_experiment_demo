"""Single follower PiPER coordinates, independent of the obsolete dual-arm task."""
import dataclasses
import math
import numpy as np
from .transforms import rgb

SCHEMA = "piper-chemical-single-v1"
ORDER = ["q1", "q2", "q3", "q4", "q5", "q6", "gripper_width"]


def validate_contract(robot):
    if robot.get("schema") != SCHEMA or robot.get("action_dim") != 7:
        raise ValueError("Expected single-arm PiPER schema with seven absolute target coordinates")
    hz = robot.get("control_hz")
    if not isinstance(hz, (float, int)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("control_hz must be finite and positive")


@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Expected finite single-arm state [7]")
        image, wrist = rgb(data["image"]), rgb(data["wrist_image"])
        result = {
            "state": state.copy(),
            "image": {"base_0_rgb": image, "left_wrist_0_rgb": np.zeros_like(image), "right_wrist_0_rgb": wrist},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise ValueError("Expected finite single-arm action sequence [horizon,7]")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < 7 or not np.isfinite(actions).all():
            raise ValueError("Invalid model action output")
        return {"actions": actions[:, :7].copy()}
