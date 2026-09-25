# NAS 训练虚拟环境

训练使用项目独立的 `.venv-train`，Python 3.11。已有 MuJoCo 环境继续复用；如需新建仿真环境，使用已有 `scripts/setup_sim_env.py`。训练环境不修改 `/openpi/.venv`。

先准备 Git 和 uv。若系统没有 uv，可按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装；有 pip 时也可执行 `python3 -m pip install --user uv`，并确保 `~/.local/bin` 在 PATH 中。

```bash
cd /mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo
python3 scripts/setup_train_env.py
source .venv-train/piper_env.sh
python -m piper_titration --help
```

脚本先检查外部 openpi 提交，然后通过其 uv.lock 安装锁定的训练与开发依赖，包含 JAX CUDA12、PyTorch、LeRobot、openpi-client，以及本项目所需的 NumPy/Pillow/PyYAML。不会分别升级这些包。最后安装本项目、检查依赖及导入，并确认 JAX 能识别 GPU。安装需网络、足够磁盘空间；驱动须由服务器提供，不在虚拟环境中安装。

若服务器 `/openpi` 版本不匹配，可用固定子模块：

```bash
git submodule update --init --recursive
python3 scripts/setup_train_env.py --openpi-root "$PWD/third_party/openpi"
source .venv-train/piper_env.sh
```

在 GPU 未分配的登录节点安装时可加 `--skip-gpu-check`，进入 GPU 节点后运行 `python -c 'import jax; print(jax.devices())'` 确认。跳过检查不表示已具备 GPU 训练能力。

脚本只重用由自身创建且来源匹配的环境；已有其他环境不被同步或清空。安装失败保留现场，修复后可重跑。若更换 openpi 来源，使用新的 `--env .venv-train-pinned`。项目中的 `.venv-*` 已被 Git 忽略。

后续执行 [烧杯微调流程](beaker_demo_training.md)时，保留激活文件设置的 `PIPER_TRAIN_PYTHON` 和 `PIPER_OPENPI_ROOT`，不要再次将解释器覆盖成原 `/openpi/.venv/bin/python`。`scripts/run.sh` 会使用这些环境变量。权重仍使用 `/mnt/cpfs/users/mrq/emboddied/openpi/checkpoints/pi05_base`。

本脚本没有执行权重加载、数据转换或训练；这些仍按微调流程验证。
