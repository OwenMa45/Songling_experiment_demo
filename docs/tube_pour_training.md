# tube_pour：同场景取试管、倒水、放回

正式 task_id 为 `tube_pour`。清单和 metadata 保持该标签，group 保持真实的 `demo_site_A`。目录名称可自定；`tube_demo` 不是注册的 task_id。旧 `test_tube_pour_return` 保留兼容已有清单，代表同类技能，不是新增第五项展示任务。

训练语言指令按用户指定原样使用：

> Pick up the tube filled with water, place it near above the beaker and pour the water, then place the tube back

成功标准：水实际倒入烧杯、无可见外溢，试管稳定放回原试管架位置。转换器从 task_id 取得该指令写入数据集，不依赖目录名称生成文本。

## 预检

同步代码后，在 songling 环境执行。预检只用 CPU；不在交互终端开启 set -e，保留进度和错误日志。

```bash
conda activate songling
cd /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo
set +e
set +u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PIPER_CONFIG="$PWD/configs/tube_demo.json"
manifest="$PWD/data/tube_pour_demo_site_A/capture_selection.json"
mkdir -p outputs/tube_demo
python -u -X faulthandler -m piper_titration --config "$PIPER_CONFIG" \
  capture-plan "$manifest" --train-only \
  > outputs/tube_demo/capture_plan.json 2> outputs/tube_demo/capture_plan.stderr.log
check_rc=$?
printf '预检退出码：%s\n' "$check_rc"
tail -n 80 outputs/tube_demo/capture_plan.stderr.log
```

清单条目数量不限于 50，新增回合需同步清单。退出码为 0 后才进行下一阶段；任务标签通过不代表其余图像、动作和验证记录一定通过。

## 转换与训练

以下命令在同一 shell 中逐条执行，每一步成功后才运行下一条。默认物理 GPU 4；使用其他已获分配设备时调整 gpu-list。

```bash
bash scripts/run.sh --gpu-list 4 convert-captures "$manifest" --train-only --repo-id local/piper_tube_demo
bash scripts/run.sh --gpu-list 4 norms
bash scripts/run.sh --gpu-list 4 train --steps 1 --batch-size 1 --experiment tube_smoke
bash scripts/run.sh --gpu-list 4 train --batch-size 1 --experiment tube_demo
```

导出训练集为 `local/piper_tube_demo_train`，receipt 为 `outputs/tube_demo/dataset_receipt.json`，模型输出放在 `outputs/tube_demo` 下。不会覆盖烧杯训练集。重复导出或训练使用新版本名称，避免覆盖已有结果。

本流程先训练独立的试管技能模型；最终若需要一个模型按语言执行 2+2 技能，应合并各任务合格回合清单，重新转换多任务训练集及计算归一化，再联合微调。本地尚未持有整批试管回合，数据质量和训练执行状态以服务器返回报告为准。
