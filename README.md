# 单臂 PiPER 化学实验 π0.5 微调

本项目负责已经采集的真机遥操数据检查、JSONL/JPEG 到 LeRobot 转换，以及 π0.5 JAX LoRA 微调和策略服务。执行实验的是单台从臂；遥操主臂不是策略的第二条机械臂。

**当前使用 songling Conda 环境与 4090 服务器，运行步骤见 [当前服务器说明](docs/server_4090.md)。** 已附带的预处理回合通过检查；其余回合须在服务器整批检查。尚未完成远程微调或生成权重。旧原始回合审查仅作为历史记录。

## 2+2 任务

| 优先级 | 任务 | 计划合格示范 |
| --- | --- | --- |
| 必做 | 抓取烧杯并移动 | 200 |
| 必做 | 取试管、倒液、放回 | 200 |
| 探索 | 滴管取液、滴液 | 200 |
| 探索 | 取倒扣滴管、反转、放入试管架孔 | 200 |

任务定义在 [chemical_tasks.json](configs/chemical_tasks.json)。200 条是采集目标，不是效果保证。当前目标是同场景 2+2 展示，支持显式 train-only；效果由同场景新执行的完整回合检验。

## 先检查数据

```bash
python scripts/audit_capture.py piper_capture_starter/data \
  --decode-images --output outputs/data_audit/audit.json
```

图像解码需要 Pillow。该命令不会连接或控制机械臂，不修改原始记录。缺少真实控制目标时，不用相邻实测状态补造 action。

## 单臂训练入口

默认配置为 [chemical_server.json](configs/chemical_server.json)，延用先前远程环境路径。状态和动作是六个关节加夹爪总开度，单位 rad/m，默认 30 Hz。π0.5 内部仍保持 32 维以兼容基础权重，模型接口仅使用前 7 维。

```bash
git submodule update --init --recursive
python scripts/doctor.py
# 复制示例，填写实际数据路径、任务 ID、采集场次/场景分组。
# 当前示例数据预检会失败，这是预期结果。
bash scripts/run.sh capture-plan configs/capture_selection.example.json
```

只有物理来源、实际控制动作、图像和成功标记通过检查后，才执行：

```bash
bash scripts/run.sh convert-captures /path/to/reviewed_selection.json
bash scripts/run.sh norms
bash scripts/run.sh train --steps 1
# 修改 training.experiment 后运行正式训练，避免覆盖单步验证。
bash scripts/run.sh train
bash scripts/run.sh serve --checkpoint /path/to/trained/step
```

转换只读原始记录，并生成数据集审查凭据；归一化和训练会检查凭据、数据集标识与频率。训练前需要完整的 JAX pi05_base 权重及匹配 openpi 版本的 Linux GPU 环境。本项目不提供实际 CAN 控制器。

操作细节和数据约定见 [单臂训练指南](docs/single_arm_training.md)。旧的 [双臂仿真说明](docs/legacy_dual_arm.md) 仅作历史参考，必须显式指定 `configs/server.yaml` 才能运行旧命令；不要用旧命令导入真机单臂数据。

## 验证范围

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

已验证数据拒绝规则、7 维映射、因果重采样和分组隔离逻辑。尚未在真实合格数据上运行 LeRobot 导出或 GPU 训练，也没有真机策略成功率结果。数据、权重和生成文件均不纳入版本控制。
