> 服务器已迁移至 4090 + songling Conda。路径、环境创建和运行命令以 [当前服务器说明](server_4090.md) 为准；本文旧 NAS 命令仅留作历史参考。

# NAS 运行准备与人工复核边界

项目：`/mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo`。
仿真环境：`/mnt/cpfs/users/mrq/emboddied/mujoco`。
训练源码：`/mnt/cpfs/users/mrq/emboddied/openpi`；默认解释器为其 `.venv/bin/python`，实际不同则设置 `PIPER_TRAIN_PYTHON`。
原始数据：项目内 `data/四项任务数据/`。用户在 NAS 运行命令并提供报告即可，无需提供服务器登录权限。

## 哪些需要人工

| 项目 | 自动处理 | 人工责任 |
| --- | --- | --- |
| 从臂身份 | 解码 CAN、比较状态数值和时间、保存日志与哈希 | 按设备安全流程安排仅从臂接入的只读验证，记录接线、设备身份、固件和验证结果；共用总线 ID 本身不能标识物理发送者 |
| 缺图 | 扫描、解码、定位丢失帧 | 先检查备份；没有原图则决定舍弃回合、分段保留或补录。程序不能恢复真实缺失画面 |
| 动作标签 | 按已核对 SDK 解码真实控制报文、组帧、转换单位、对齐并生成 candidate_action | 确认报文对应执行从臂的目标，核实控制模式和时序；不需要人工逐帧填写七个数 |
| 成功标注 | 检查标签是否存在、统计和筛选 | 初期逐回合观看视频并核实任务结果，填写成功/失败和原因；末端到位不足以证明倒液或吸液成功 |

身份验证按硬件配置进行；接线、固件、ID 或转发链路变化后应复核，不必每回合拆线。每回合引用适用的验证记录。保留原始文件，把验证记录和数据修复结果作为派生资料；不能只改 false 为 true 绕过检查。当前工具没有将候选 CAN 动作自动批准为训练动作的功能。

分段也不能把断开的时间片拼成连续轨迹；目标任务如要求完整抓取、移动、放置，缺失关键过程的回合应补录。当前样例仍有 null action、缺失 front/000060.jpg 和未审核成功标签。

## 必需的基础权重

本项目使用 **JAX/Orbax 格式的 pi05_base 完整基础检查点**，官方地址：

```text
gs://openpi-assets/checkpoints/pi05_base
```

默认本地布局：

```text
/mnt/cpfs/users/mrq/emboddied/openpi/checkpoints/pi05_base/
  params/        # 完整参数树、元数据及全部分片
  ...            # 下载包中的其他文件原样保留
```

`PIPER_CHECKPOINT` 指向 `pi05_base` 根目录，训练入口会追加 `/params`。PyTorch 的单个 safetensors、LoRA 适配器或空 params 目录不能代替此检查点。无需另行拼装 Gemma/SigLIP 基础权重。数据归一化统计由本项目在合格训练集上计算；不套用其他机器人的统计。

可使用现有 openpi 的下载器，下载位置由其缓存管理；下面会自动把实际路径传给本项目。此命令会下载大文件，准备充足磁盘空间，并在可访问官方存储的服务器上执行：

```bash
cd /mnt/cpfs/users/mrq/emboddied/openpi
export PIPER_TRAIN_PYTHON=/mnt/cpfs/users/mrq/emboddied/openpi/.venv/bin/python
export OPENPI_DATA_HOME=/mnt/cpfs/users/mrq/emboddied/openpi/.cache/openpi
export PIPER_CHECKPOINT="$($PIPER_TRAIN_PYTHON -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))')"
"$PIPER_TRAIN_PYTHON" -c 'from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(); print("tokenizer cached")'
printf '%s\n' "$PIPER_CHECKPOINT"
```

tokenizer 会额外缓存 `gs://big_vision/paligemma_tokenizer.model`；离线训练节点需预先准备同一缓存。上述下载不会把文件放到默认 checkpoints 路径，故应在后续运行的 shell 中保留 `PIPER_CHECKPOINT`；已有完整权重时直接设置此变量到已有根目录即可。

## 上传代码后的检查

```bash
cd /mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo
git submodule update --init --recursive
mkdir -p outputs/data_audit
bash scripts/run.sh doctor > outputs/doctor.json
"$PIPER_TRAIN_PYTHON" scripts/audit_capture.py "data/四项任务数据" \
  --decode-images --output outputs/data_audit/nas_audit.json
```

审查遇到不合格回合会返回非零，请阅读报告而非跳过错误。doctor 也包含历史仿真依赖检查；无渲染需求时不用 `--render`。它对权重只做目录结构检查，实际兼容性需要后续单步训练验证。

本项目要求 openpi 提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。服务器现有版本不同会停止训练；可选择本项目固定的 `third_party/openpi` 并按其锁文件准备独立训练环境，不要直接重置已有 openpi 工作目录。对应设置 `PIPER_OPENPI_ROOT` 和 `PIPER_TRAIN_PYTHON`。

依赖由对应 openpi 锁文件管理（JAX、Flax、Orbax、LeRobot 等），审查需要 Pillow。不要单独升级 JAX、Torch 或 LeRobot。官方给出的 LoRA 显存参考为大于 22.5 GB；本项目能否在实际 GPU 上训练仍以单步实测为准。MuJoCo 不参与真实数据的梯度训练。

全部回合复核后建立选择清单（含 task_id 和真实采集场次/场景 group），按 [单臂训练文档](single_arm_training.md) 执行 `capture-plan`、`convert-captures`、`norms` 和 `train --steps 1`，确认恢复权重、反向传播、保存检查点成功，再使用新 experiment 开始正式训练。

权重和环境参考：[固定版本 openpi 官方说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)。
