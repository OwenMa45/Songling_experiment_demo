# External sources

Both directories are Git submodules. Initialize with `git submodule update --init --recursive`.
Use recorded gitlinks, not `--remote`, for reproducible runs. Retain upstream licenses.

| Dependency | Pinned commit | Usage |
| --- | --- | --- |
| piper_ros | ac41fcbcdda598f01b51cf6175ed9a24d0dacadc | MJCF, inertias and meshes only |
| openpi | 215abfb217dbac7d5f1273282331b9b1866c0479 | Training API and inference protocol |

The configured external server checkout of openpi must match the pinned commit.
No ROS, Gazebo or mujoco-py runtime is required. Generated MJCF adaptations live
under outputs, never inside these submodules.
