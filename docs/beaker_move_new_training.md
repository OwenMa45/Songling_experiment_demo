# beaker_move_new 新数据训练

沿用 20260928_120904_824429 已完成的烧杯训练：从 pi05_base 做 π0.5 JAX LoRA，4×3090，batch=4，FSDP=4，action_horizon=16，10000 步，seed=42。正式训练从基础权重开始，不接着旧烧杯或 smoke 检查点训练。

## 独立配置

- 数据包：`data/beaker_new/`，选择清单：`capture_selection.json`。
- 任务 ID：`beaker_move_new`，保留采集清单和元数据的原始标签。
- 指令：`Pick up the beaker, place it in the middle of the table, open the gripper to release the beaker, then return the robot arm to its starting position.`
- 配置：`configs/beaker_move_new.json`。
- 编排配置：`configs/beaker_move_new_profile.json`。
- LeRobot repo_id：`local/piper_beaker_move_new_train`。
- 转换凭据：`outputs/beaker_move_new/dataset_receipt.json`。
- 归一化与检查点：`outputs/beaker_move_new/` 下的 `training_assets/` 和 `checkpoints/`。

旧烧杯、倒液数据及模型保持独立。`--task both` 仍只运行原来的两个任务，新模型必须显式指定 `--task beaker_move_new` 和新 profile。

## 上传完成后运行

清单有 228 条，预处理报告为审核 232 条、通过 228 条、排除 4 条。报告不等于训练侧已通过检查。

确保整个数据包传完，尤其是全部回合的 metadata、samples、双相机图片，以及 metadata 引用的 `verification/`。保持真实单场景 group，使用原有 train-only 流程。

在 NAS 项目根目录启动（SSH 连接方式沿用原入口）：

```bash
cd /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo
python scripts/train_two_skills.py \
  --task beaker_move_new \
  --profile configs/beaker_move_new_profile.json \
  --stage all --steps 10000 \
  --batch-size 4 --fsdp-devices 4 \
  --experiment beaker_move_new_pipeline
```

先确认节点 10.71.106.251 的 GPU 0–3 可用。可在上述命令末尾添加 `--dry-run` 只预览；长训练建议放在已有 tmux 会话中运行。`--experiment` 在这个编排入口用于日志命名，重复运行应换一个日志名；实际 smoke/train 实验名由任务名和 UTC 时间自动生成。

流程依次执行：逐回合完整审核及图像解码、LeRobot 导出、独立归一化与坐标核对、1 步 smoke、10000 步正式训练。任何环节失败即停止，不静默丢掉缺失回合。已有合格 receipt 和 norms 会验证后复用。

每次运行记录在 `outputs/two_skills/<UTC时间>/run.json`，取 `tasks.beaker_move_new.train_checkpoint` 作为正式模型；不要使用 `smoke_checkpoint`。正式检查点目录形如：

```text
outputs/beaker_move_new/checkpoints/pi05_piper_chemical_lora/beaker_move_new_train_<UTC时间>/9999
```

完成后用 run.json 记录的 config 与 train_checkpoint 启动现有 serve 入口。训练损失只用于判断优化过程，demo 效果仍需真机完整回合验证。

## 已完成的环境检查

训练节点现有 songling 环境可识别 4 张 3090；JAX、Flax、Orbax、LeRobot、openpi 依赖和固定 openpi 提交通过 doctor。基础权重结构检查通过，实际恢复、反向传播和保存仍待上传完成后的 smoke 验证。远程检查报告：`outputs/beaker_move_new/doctor.json`。

## 2026-10-02 本次运行

用户确认上传完成后，已逐条核验 228 条回合的 metadata/samples 文件存在、任务标签和语言指令一致，以及全部审核文件引用的 SHA-256。元数据声明合计 92782 帧。完整图像解码和内容审核由转换入口继续执行，基础文件检查不代表全量审核已经通过。

22:43（北京时间）已在 10.71.106.251 的独立 tmux 会话启动完整流程：

- tmux：`beaker_move_new_20261002_144346`
- 日志：`outputs/remote_logs/beaker_move_new_20261002_144346.log`
- 启动记录与脚本：`outputs/beaker_move_new/pipeline_20261002_144346/`
- 流程状态：`outputs/two_skills/20261002_144354_416614/run.json`
- 退出码：`outputs/beaker_move_new/pipeline_20261002_144346/exit_code`，进程退出后生成。

该流程会依次完成转换、归一化、smoke 和正式训练，失败即停；不要重复启动同一个数据集的转换。此处记录的是启动信息，最终结果以 run.json 中的 completed 状态、正式 train_checkpoint 和退出码 0 为准。
