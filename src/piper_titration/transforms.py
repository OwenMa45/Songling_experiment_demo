"""Shared train/inference mapping; PiPER coordinates have no ALOHA conversions."""
import dataclasses
import numpy as np


def rgb(value):
    a = np.asarray(value)
    if a.ndim != 3:
        raise ValueError("Expected rank-three RGB image")
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] != 3:
        raise ValueError("Expected three RGB channels")
    if np.issubdtype(a.dtype, np.floating):
        if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
            raise ValueError("Float RGB must be finite in [0,1]; use uint8 for [0,255]")
        a = (a*255).astype(np.uint8)
    elif a.dtype != np.uint8:
        raise ValueError("RGB must be uint8 or float [0,1]")
    return a


@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, data):
        image, wrist = rgb(data["image"]), rgb(data["wrist_image"])
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError("PiPER state must have 14 coordinates")
        result = {"state": state, "image": {"base_0_rgb": image, "left_wrist_0_rgb": np.zeros_like(image), "right_wrist_0_rgb": wrist}, "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.True_}}
        for key in ("prompt", "actions"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :14]}
