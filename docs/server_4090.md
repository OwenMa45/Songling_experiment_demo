# 4090 服务器：当前运行入口

本文件替代旧阿里 PPU/NAS 的路径和虚拟环境说明。当前任务为同场景 2+2 技能展示，使用真机数据离线微调，无需 MuJoCo。

| 用途 | 路径 |
| --- | --- |
| 项目 | /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo |
| openpi | /mnt/nas/share/home/lzk/mrq/robot/third_party/openpi |
| 权重根目录 | /mnt/nas/share/home/lzk/mrq/robot/openpi_data/openpi-assets/checkpoints/pi05_base |
| 原始/预处理数据 | 项目 data/，可添加 tube_pour 等目录 |
| LeRobot 数据 | 项目 data/lerobot/ |
| 训练输出 | 项目 outputs/beaker_demo/ 或 outputs/chemical/ |
| openpi 缓存 | /mnt/nas/share/home/lzk/mrq/robot/openpi_data |
| Hugging Face 缓存 | /mnt/nas/share/home/lzk/mrq/robot/cache/huggingface |

Conda 环境为 songling。配置中 @current 表示实际运行进程的 Python，不猜测 Conda 安装路径。run.sh 默认选择 songling，也可用 PIPER_TRAIN_PYTHON 指定解释器。OPENPI_DATA_HOME 复用已有权重缓存；HF_HOME、HF_LEROBOT_HOME 均可环境覆盖。

## 创建训练环境（尚未创建时）

```bash
conda create -n songling python=3.11 -y
conda activate songling
python -m pip install uv
cd /mnt/nas/share/home/lzk/mrq/robot/third_party/openpi
git rev-parse HEAD
# 必须为 215abfb217dbac7d5f1273282331b9b1866c0479
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact
```

已配置好的环境先诊断，按报告补缺失依赖，不单独升级 JAX/Torch/LeRobot。uv 使用 openpi 锁文件；--inexact 保留 Conda 环境已有额外包。此步骤由用户在服务器执行，本地没有安装或修改远程环境。

## 烧杯微调

把完整数据包放在 data/beaker_move_demo_site_A（若实际目录不同，修改下面 manifest）。后续 tube_pour 等目录不会自动决定任务语义，仍在清单填写 configs/chemical_tasks.json 中的 task_id。

```bash
set -euo pipefail
cd /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo
unset PIPER_TRAIN_PYTHON PIPER_OPENPI_ROOT PIPER_CHECKPOINT
export PIPER_CONFIG="$PWD/configs/beaker_demo.json"
manifest="$PWD/data/beaker_move_demo_site_A/capture_selection.json"
mkdir -p outputs/beaker_demo
bash scripts/run.sh --gpu-list 4 doctor > outputs/beaker_demo/doctor.json
bash scripts/run.sh --gpu-list 4 capture-plan "$manifest" --train-only > outputs/beaker_demo/capture_plan.json
bash scripts/run.sh --gpu-list 4 convert-captures "$manifest" --train-only --repo-id local/piper_beaker_demo
bash scripts/run.sh --gpu-list 4 norms
bash scripts/run.sh --gpu-list 4 train --steps 1 --batch-size 1 --experiment beaker_smoke
# 上一步成功后再运行：
bash scripts/run.sh --gpu-list 4 train --batch-size 1 --experiment beaker_demo
```

doctor 在单臂配置下默认只检查训练相关资源，并实际尝试 jax.devices()，不要求 ROS/MuJoCo。提供的 nvidia-smi 显示驱动 550.144.03、CUDA 标示 12.4；该标示不是 Python 环境 CUDA/JAX 安装证明。权重目录检查也不是实际恢复证明，以单步训练为准。

## 使用多卡

run.sh 默认只使用物理 GPU 4。--gpu-list 可放在命令前后，只允许授权的 4、5、6、7，拒绝重复编号。传入的设备在进程内从 0 重新编号。不会自动占满服务器八张卡。

```bash
bash scripts/run.sh --gpu-list 4,5,6,7 train --steps 1 --batch-size 4 --experiment beaker_smoke_4gpu
bash scripts/run.sh --gpu-list 4,5,6,7 train --batch-size 4 --experiment beaker_demo_4gpu
```

使用一个 JAX 进程，不使用 torchrun。默认 fsdp_devices=1，四卡为数据并行、每卡模型副本；不能理解为四卡显存自动合并成 96GB。batch-size 是全局批量，须能被可见设备数整除。可显式传入 `--fsdp-devices 4` 启用四卡参数分片，设备数必须能被此值整除。

## 2026-09-27 单卡初始化 OOM

用户日志已证明数据、归一化资产读取及 12.5 GiB 基础参数恢复成功，但初始化训练状态时额外 1.12 GiB 分配失败，尚未完成梯度更新。ROCm/TPU 后端探测提示不是本次根因。日志未记录显存比例或同时运行的进程，不能断言仅由默认比例导致。

确认目标卡没有其他大显存任务后，可先在新进程中设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`、`XLA_PYTHON_CLIENT_PREALLOCATE=true` 重试单卡，并使用新的 experiment。不要重新转换数据或计算 norms。参考 [JAX 显存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。

仍不足时，同步代码后尝试授权卡上的 FSDP：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=true \
bash scripts/run.sh --gpu-list 4,5,6,7 train --steps 1 \
  --batch-size 4 --fsdp-devices 4 --experiment beaker_smoke_fsdp4
```

FSDP 减少可分片的参数和训练状态占用，但固定版本上游初始化仍以 replicated 输入加载基础参数，因此不保证消除初始化峰值。当前仅通过布局校验测试，多卡运行需要服务器实测。保留 PIPER_CONFIG、PIPER_DATASET_RECEIPT、HF_LEROBOT_HOME 与成功 norms 时一致；已有失败目录保留，重试用新名称。

已有 dataset receipt 或训练目录不会被覆盖；重试需选择新实验名，重新导出需同时更改 repo-id 与配置/receipt 路径。
