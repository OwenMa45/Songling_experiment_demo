# 双 PiPER 化学滴定仿真

> **2026-09-24 数据审查更新：**项目正转向单臂真机遥操多任务微调。已提供的回合属于仅观测记录，703 帧 action 全为空，暂不能用旧双臂训练入口微调。检查结果和 2+2 任务采集安排见 [数据审查报告](docs/capture_audit_20260924.md)，批量检查使用 `scripts/audit_capture.py`。下文仍为原双臂实现说明。

MuJoCo 双臂演示、示范采集与回放、LeRobot 数据转换，以及 π0.5 JAX LoRA 训练/推理适配。
固定试管，持管臂握住预装液滴管，夹捏臂挤压胶头，独立计数模块达到目标后停止。

**当前状态：代码已实现；本地通过纯逻辑、数据接口及运动学测试。MuJoCo 动力学、EGL 渲染和 GPU 训练尚未实测，不能据此宣称已达到 90% 成功率或可直接部署真机。**

## 服务器开始运行

先初始化外部模型与 openpi 的固定版本：

```bash
git submodule update --init --recursive
python scripts/doctor.py --render
```

诊断不会安装包或下载模型。默认使用：

- 仿真：`/mnt/cpfs/users/mrq/emboddied/mujoco/bin/python`
- openpi：`/mnt/cpfs/users/mrq/emboddied/openpi`
- 训练 Python：`/mnt/cpfs/users/mrq/emboddied/openpi/.venv/bin/python`
- JAX 权重：`/mnt/cpfs/users/mrq/emboddied/openpi/checkpoints/pi05_base`

依赖缺失时，按 [远程运行指南](docs/remote.md) 安装；所有默认值集中在 [服务器配置](configs/server.yaml)。

```bash
bash scripts/run.sh build
bash scripts/run.sh demo --target 3 --output outputs/demo_001 --record --video
bash scripts/run.sh replay outputs/demo_001/episode
bash scripts/run.sh evaluate --episodes 100 --output outputs/expert_eval
```

每次运行使用新的输出目录，不覆盖既有数据。`demo` 失败返回非零退出码；`evaluate` 只有完成至少 100 个固定场景回合且成功率达到 90% 才返回零。

## 数据与训练

```bash
# 可先采集少量回合检查数据，随后增加规模；只有成功回合进入训练集。
bash scripts/run.sh evaluate --episodes 20 --randomized --record --output outputs/demonstrations
bash scripts/run.sh convert outputs/demonstrations --repo-id local/piper_titration
bash scripts/run.sh norms
bash scripts/run.sh train --steps 1
```

20 回合用于数据链路冒烟验证，并非充分的训练数据量；少于 100 回合的 `evaluate` 不报告验收通过。
正式训练应更换配置中的 `training.experiment`，避免与单步验证目录冲突。

```bash
bash scripts/run.sh train
# 将路径替换成实际训练生成的 step 目录
bash scripts/run.sh serve --checkpoint outputs/checkpoints/pi05_piper_lora/titration_lora/9999
# 在另一个终端执行
bash scripts/run.sh demo --target 3 --output outputs/policy_demo --video --policy-uri ws://127.0.0.1:8000
```

详细依赖、权重准备和单卡运行安排见 [远程运行指南](docs/remote.md)。
数据字段、仿真假设和真机校准要求见 [接口与建模说明](docs/interfaces.md)。
外部代码由 [third_party](third_party/README.md) 中的 submodule 管理。

## 测试

```bash
PYTHONPATH=src /mnt/cpfs/users/mrq/emboddied/mujoco/bin/python -m unittest discover -s tests -v
```

纯逻辑测试不需要 GPU；MuJoCo 未安装时会跳过动力学测试。完整验证还需运行 EGL 诊断、带视频演示、回放及 100 回合评估。
