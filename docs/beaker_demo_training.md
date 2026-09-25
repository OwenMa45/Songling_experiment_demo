# 同场景 2+2 技能展示：第一阶段烧杯微调

验收目标是采集场景内的两项必做技能（烧杯移动、取试管倒液放回）与两项探索技能（滴管取液滴液、倒扣滴管反转放入孔）。不要求跨场景泛化。使用显式 `--train-only` 保留真实 group，不虚构分组或验证集。训练损失用于检查优化过程；效果通过同场景下新执行的完整回合评估，分别记录各技能成功/失败及原因。

## 当前数据判断

数据包声明 20 条审核回合，19 条成功，1 条因内部保持超出 1000 ms 被排除。清单均为 beaker_move/demo_site_A。当前本地只附有 episode_20260924_182612，不能据汇总表宣称另外 18 条已经通过独立审查。

附带回合通过 prepared-v2 预检：647 帧、约 30 Hz、1294 张图像；身份和保持行为记录 SHA-256 匹配。先前已逐帧与原始 CAN 交叉核对动作。以用户提供的通过记录作为硬件验证声明，不宣称本项目重新进行了硬件测试。原始 hold 日志和人工审核 CSV 建议随数据归档；当前视觉字段含 assume 标记，实际演示前需复核视频是否覆盖完整成功过程。

可进入固定场景微调试跑。NAS 上必须对清单全部 19 条做预检；少任何回合都会报错，不会静默跳过或用本地唯一回合替代整批。负夹爪坐标原样保留，推理执行端必须采用相同约定。

## NAS 执行

将整个 beaker_move_demo_site_A 数据包（包括 verification、episodes、清单与汇总）放到：
`/mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo/data/四项任务数据/beaker_move_demo_site_A`。
代码使用 Git 同步；数据被 .gitignore 排除，需单独传输。

在 NAS 同一个 shell 中按顺序执行，任何步骤失败先处理错误：

```bash
set -euo pipefail
cd /mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo
export PIPER_CONFIG="$PWD/configs/beaker_demo.json"
export PIPER_TRAIN_PYTHON="${PIPER_TRAIN_PYTHON:-/mnt/cpfs/users/mrq/emboddied/openpi/.venv/bin/python}"
export PIPER_CHECKPOINT=/mnt/cpfs/users/mrq/emboddied/openpi/checkpoints/pi05_base
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_LEROBOT_HOME="$PWD/data/lerobot"
manifest="$PWD/data/四项任务数据/beaker_move_demo_site_A/capture_selection.json"
mkdir -p outputs/beaker_demo

# 完整回合预检；直接使用训练环境，无需 MuJoCo。
"$PIPER_TRAIN_PYTHON" -m piper_titration --config "$PIPER_CONFIG" \
  capture-plan "$manifest" --train-only > outputs/beaker_demo/capture_plan.json

# 创建训练集，不创建空验证/测试集；已存在的导出不会覆盖。
bash scripts/run.sh convert-captures "$manifest" --train-only \
  --repo-id local/piper_beaker_demo
bash scripts/run.sh norms

# 一步验证权重恢复、反向传播和检查点保存。
bash scripts/run.sh train --steps 1 --experiment beaker_smoke

# 确认上一步成功后，从基础权重开始正式试跑；独立实验目录。
bash scripts/run.sh train --experiment beaker_demo
```

默认正式试跑为 10000 步、batch=1、LoRA，属于起始设置而非已验证的最佳训练时长。19 条同场景示范应结合真实执行效果选择训练时长，不能把低训练损失当作技能成功。smoke 检查点不作为正式结果，正式命令不接着单步检查点训练。

openpi 必须匹配项目固定提交；本项目不会修改服务器已有源码。如版本不一致，按 NAS 文档选择固定子模块和相应锁定环境。权重目录存在不等于实际恢复已通过。全部转换、训练、推理尚需在 NAS 实测。

若重复运行遇到已有数据集/检查点，请使用新 repo-id（同时更新配置）、新 receipt 路径和新 experiment；不删除旧结果。新增另外三项技能后建立多任务清单，以各自 task_id 生成语言指令，并使用新数据集名称重新转换和计算归一化；同场景仍可使用 train-only。
