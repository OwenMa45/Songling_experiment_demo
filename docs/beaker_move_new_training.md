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

## 2026-10-03 归一化修复与续跑

上一轮 228 条回合全部通过审核，数据转换成功，因果重采样后为 92662 帧。首次 `norm-check` 报缺少 `piper_norm_audit.json`，属于新数据集的预期情况，编排随后尝试计算统计。

实际中断在 `norms` 的第一个 batch：统计使用 batch=1 以覆盖所有帧，但上游加载器默认在全部 4 个 JAX GPU 设备上分片，导致 `(1,16,7)` 首维无法被 4 整除。未进入 smoke 或正式训练，也未成功生成归一化统计。

修复在 CLI 导入项目/openpi 模块前为 `norms` 进程设置 `JAX_PLATFORMS=cpu`，保持 batch=1、全量统计和既有坐标审计。后续训练是独立进程，继续使用 4×3090、batch=4、FSDP=4。新增进程级回归测试已在训练服务器通过：原报错形状可放置到 CPU，训练命令保留请求的 GPU 后端。

已复用现有 receipt 启动续跑，不重新导出数据：

- tmux：`beaker_move_new_normfix_20261003_023521`
- 日志：`outputs/remote_logs/beaker_move_new_normfix_20261003_023521.log`
- 启动记录与退出码：`outputs/beaker_move_new/pipeline_20261003_023521/`

本次直接使用 songling 的 Python，避开无关的 Conda libmamba 插件启动告警。旧失败目录保留。最终完成与否仍以本次 run.json、正式检查点及退出码为准。

## 2026-10-03 检查点内存修复后重新训练

归一化修复后的训练在保存第 1000 步时被 Linux global OOM 杀死（PID 174296，anon-rss 约 39.4 GiB）。该步仅有 Orbax 临时目录，未用作恢复点。

项目训练适配层现在使用同步检查点保存，并对 params 和 train_state 的 PyTree handler 分别设置 `save_concurrent_gb=2`，降低写入并发及保存与计算重叠造成的内存压力。保持上游文件、参数和训练状态格式不变；真实 composite 保存/恢复回归测试已通过。这不是整个训练进程的 4 GB 内存上限，也不保证消除共享主机的所有 OOM 风险。

17:54（北京时间）启动新流程，复用数据与归一化，先验证完整模型单步保存，再从基础权重开始 10000 步正式训练：

- tmux：`beaker_move_new_memfix_20261003_095418`
- 日志：`outputs/remote_logs/beaker_move_new_memfix_20261003_095418.log`
- 启动记录与退出码：`outputs/beaker_move_new/pipeline_20261003_095418/`

失败目录保留，不把临时检查点重命名为完成检查点；新模型完成状态以此次运行记录为准。
