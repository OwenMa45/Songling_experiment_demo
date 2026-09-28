"""Single follower PiPER coordinates, independent of the obsolete dual-arm task."""
import dataclasses
import math
import numpy as np
from .transforms import rgb

SCHEMA = "piper-chemical-single-v1"
ORDER = ["q1", "q2", "q3", "q4", "q5", "q6", "gripper_width"]
DELTA_MASK = (True, True, True, True, True, True, False)


def coordinate_contract(robot, horizon):
    validate_contract(robot)
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    return {
        "version":1, "schema":SCHEMA, "state_order":ORDER,
        "units":["rad"]*6+["m"], "control_hz":robot["control_hz"],
        "action_horizon":horizon, "model_action_dim":32,
        "coordinate_frame":"follower SDK joint coordinates; no Cartesian pose or base/world transform",
        "joint_conversion":"identity: retain SDK zero/sign; no degree conversion, wrap or ALOHA remapping",
        "gripper_conversion":"identity: signed SDK coordinate in metres; no clipping, 0..1 conversion or inversion",
        "dataset_actions":"absolute CAN position targets",
        "delta_mask":list(DELTA_MASK),
        "delta_reference":"each chunk target minus the SAME observation state at chunk origin, not consecutive differences",
        "normalization":"pi05 q01/q99 with epsilon 1e-6; after delta conversion, before 32D zero padding",
        "policy_output":"absolute targets rad/metres; unnormalize then add request state ONCE",
        "images":{"image":"front RGB -> base_0_rgb", "wrist_image":"wrist RGB -> right_wrist_0_rgb",
                  "unused":"left_wrist_0_rgb masked", "resize":"224x224 aspect-preserving padding",
                  "encoding":"uint8 HWC/CHW 0..255 or float HWC/CHW 0..1"},
    }


def data_transforms():
    from openpi import transforms as t
    return t.Group(inputs=[Inputs()], outputs=[Outputs()]).push(
        inputs=[t.DeltaActions(DELTA_MASK)], outputs=[t.AbsoluteActions(DELTA_MASK)])


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
