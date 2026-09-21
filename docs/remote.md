# 远程环境、权重和运行顺序

## 1. 环境边界

本项目不使用本地 Windows 作为目标运行环境。Python 3.11/3.12 Linux 是建议配置；训练依赖以固定 openpi 提交的 `uv.lock` 为准。仿真与训练使用不同解释器，通过 websocket 交换观测和动作。

`scripts/run.sh` 根据子命令选解释器，不会安装任何依赖。`PIPER_SIM_PYTHON`、`PIPER_TRAIN_PYTHON` 可覆盖解释器；`PIPER_OPENPI_ROOT`、`PIPER_CHECKPOINT`、`PIPER_PIPER_ROOT`、`PIPER_OUTPUT` 可覆盖资源位置。`PIPER_CONFIG` 可选择另外一份 YAML。相对配置路径按项目根目录解析；命令行 `--output` 按当前工作目录解析。

如果只改 YAML 中的解释器位置，通过对应 Python 直接运行 `python -m piper_titration --config ...`，或同时设置启动脚本的 `PIPER_*_PYTHON` 环境变量。启动脚本不自行解析 YAML。

## 2. 检查与安装

在项目根目录执行：

```bash
git submodule update --init --recursive
python scripts/doctor.py --render
```

诊断输出每项状态和缺失原因，并在存在失败项时返回非零。检查包括两个解释器、包版本、GPU、两份外部代码提交、JAX `params` 目录、单臂模型加载与 EGL 图像渲染。权重目录检查不能代替实际恢复权重。

需要的仿真包为 `mujoco>=3.2,<4`、`numpy>=1.26,<2`、`PyYAML>=6`、`imageio>=2.34`、`imageio-ffmpeg>=0.5`，以及所选 openpi 源码内的 `openpi-client`。当前逆运动学只依赖 NumPy，无需另装 SciPy。测试可用标准库 unittest，pytest 是可选开发依赖。

```bash
SIM=/mnt/cpfs/users/mrq/emboddied/mujoco/bin/python
OPENPI=/mnt/cpfs/users/mrq/emboddied/openpi
"$SIM" -m pip install -e .
"$SIM" -m pip install -e "$OPENPI/packages/openpi-client"
```

训练环境沿用 openpi 的锁文件。若已有工作环境，先诊断，不重新同步或升级。仅在尚未准备该环境时，根据其 README 在 openpi 目录执行 `GIT_LFS_SKIP_SMUDGE=1 uv sync` 和 `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`。本项目训练入口也需要 PyYAML；缺失时仅在该解释器中安装 `PyYAML>=6`。

不要在仿真环境安装整套 openpi，也不要单独升级训练环境的 JAX、Torch 或 LeRobot。无需 ROS、Gazebo、mujoco-py、TensorFlow 或真实机械臂 SDK。

Linux 无窗口渲染默认 `MUJOCO_GL=egl`。服务器需可访问 NVIDIA GPU、驱动提供的 EGL 库及设备节点；容器需暴露 GPU 和图形驱动能力。若诊断报告 `libEGL`/驱动缺失，应由服务器环境管理员按发行版补齐，项目不自动执行 apt 或修改驱动。`MUJOCO_GL=osmesa` 仅作为具备 OSMesa 的 CPU 渲染替代，不能用它冒充 EGL 检查通过。

## 3. 固定版本与已有服务器 checkout

- PiPER：`ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`
- openpi：`215abfb217dbac7d5f1273282331b9b1866c0479`

本项目没有修改这两个上游仓库。已有 openpi 若版本不同，训练入口会明确拒绝。可选择本项目固定版本并为其准备匹配的环境：

```bash
export PIPER_OPENPI_ROOT="$PWD/third_party/openpi"
export PIPER_TRAIN_PYTHON="$PWD/third_party/openpi/.venv/bin/python"
```

不要直接重置含有自己改动的服务器 checkout。若要适配其他 openpi 版本，应同时更新 gitlink、版本检查和接口测试。

## 4. 下载什么权重

准备官方 **π0.5 base 的 JAX 完整检查点**：

```text
gs://openpi-assets/checkpoints/pi05_base
```

将该目录的完整内容保存到：

```text
/mnt/cpfs/users/mrq/emboddied/openpi/checkpoints/pi05_base/
└── params/
    └── ... 原始 Orbax/TensorStore 文件，保持完整结构
```

也可设置 `PIPER_CHECKPOINT` 指向实际目录。不要仅下载单个权重文件，也不要将 PyTorch 的 `model.safetensors` 当成 JAX 检查点。

默认配置采用 `gemma_2b_lora` 和 `gemma_300m_lora`，动作内部维度保留 32 以兼容基础权重，PiPER 使用前 14 维。无需先准备 ALOHA、DROID 或 LIBERO 专用权重。

openpi 还会加载 `gs://big_vision/paligemma_tokenizer.model`。离线服务器需在联网环境中使用相同 openpi 的下载器提前缓存 tokenizer，再迁移对应缓存；仅下载机器人权重并不足以保证完全离线启动。缓存位置由 openpi 的 `OPENPI_DATA_HOME` 控制。

## 5. 数据到策略

1. 先完成 `build`、脚本 `demo --video --record` 和 `replay`，检查操作位置、相机遮挡与停止后的液滴。
2. 用 `evaluate --record` 采集示范。转换器递归读取回合，过滤失败回合，按整回合分为 80% 训练、20% 测试，至少需要两条成功示范。
3. `convert --repo-id local/piper_titration` 生成 `local/piper_titration_train` 和 `local/piper_titration_test`，目录使用 LeRobot 的 `HF_LEROBOT_HOME`。写出 `split.json`，不上传 Hugging Face，不覆盖已有数据集。
4. `norms` 仅从训练集计算关节增量和夹爪绝对开度的归一化统计。
5. `train --steps 1` 验证实际加载、反向传播与保存。输出在 `outputs/checkpoints/pi05_piper_lora/<experiment>/0`。这一步需要完整权重和 GPU。
6. 修改 `training.experiment` 后进行正式训练。默认 batch=1、动作序列长度=16、10000 步、W&B 关闭；这些是起始配置，不代表达到效果所需的最优训练量。
7. `serve --checkpoint <实际step目录>` 加载训练权重及随检查点保存的归一化统计。默认只监听本机 127.0.0.1:8000。
8. 使用 `demo/evaluate --policy-uri ws://127.0.0.1:8000` 做闭环评估；与脚本基线采用相同固定/随机化设置。

训练会在实验目录保存 `piper_run.json`，服务启动时使用其中的训练配置与控制频率，避免当前 YAML 的后续改动改变已有策略的接口。迁移检查点时一并保留此文件及 step 目录中的 assets。

24GB GPU 显存紧张。训练时停止仿真采集与策略服务；只在评估时同时运行推理服务和小分辨率渲染。先以 batch=1 验证实际显存需求。如果仍 OOM，不能将理论 LoRA 门槛视为保证，需要降低序列长度或提供更大显存环境。中断不会自动恢复或覆盖已有训练目录。

## 6. 验收与未验证部分

固定场景 100 回合，目标滴数轮换 3/5/10；成功要求准确入管、无外溢、无检测到的非预期碰撞、无超时，并完成停止后的 1.5 秒观察。随机化报告单独输出，不计为固定场景验收。

当前本地通过逻辑/数据/运动学检查；本地安装 MuJoCo 的请求未获批准，因而尚未执行动力学闭环、视频视觉检查、EGL、LeRobot 实际转换或 GPU 训练。仓库提供相应测试和命令，尚无 100 回合成功率结果。
