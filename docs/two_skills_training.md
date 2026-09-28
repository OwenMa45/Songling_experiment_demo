# 固定场景两个独立技能模型

目标仅为 beaker_move 与 tube_pour，不要求跨场景泛化，也不训练联合多任务模型。两者各自从 pi05_base 做 LoRA，独立数据、归一化资产、检查点。训练完一个后释放进程，再训练另一个；不在四张卡上同时运行两套训练。

NAS 入口：`scripts/train_two_skills.py`。复用 SSH 主机 10.71.106.251、用户 lzk、私钥 /home/lzk/.ssh/id_ed25519、songling 环境、4×3090、batch=4、FSDP=4。远程路径仍假设共享 NAS 同路径挂载。--host/--user/--identity/--python/--conda/--project/--gpu-list 等连接与硬件覆盖参数与原 launcher 一致。

## 数据配置

编辑 `configs/two_skills.json`（所有路径指远端，默认相对于项目目录）：

| 任务 | 配置 | 默认转换凭据 | 无凭据时的原始选择清单 |
| --- | --- | --- | --- |
| beaker_move | configs/beaker_demo.json | outputs/chemical/dataset_receipt.json | data/beaker_move_demo_site_A/capture_selection.json |
| tube_pour | configs/tube_demo.json | outputs/tube_demo/dataset_receipt.json | data/tube_pour/capture_selection.json |

烧杯沿用已知实际转换凭据。倒液凭据位置和目录请按服务器实际情况填写，不假定已验证存在。如果两个数据集都已经转换，只需指定现有凭据，原始清单不会被读取；数据集 repo_id 必须与各自配置相符。不会自动重命名、移动、修改凭据或把两个任务合并。旧数据集特征/坐标契约不兼容时会停止，不能为避免重新转换而略过校验。

## 运行

```bash
cd /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo
# 预览远程命令，无连接、无训练
python scripts/train_two_skills.py --stage all --dry-run

# 一键顺序处理两个任务：检查/必要时转换、检查/必要时计算norms、单步、正式训练
python scripts/train_two_skills.py --stage all --steps 10000
```

默认 --task both。已转换的数据通过凭据、特征和任务标签检查后复用；已绑定且可用的归一化通过 norm-check 后复用，不重复计算。norm-check 失败时 prepare/all 会尝试重算该任务统计，若失败则停止，不继续训练。

也可分阶段运行：

```bash
python scripts/train_two_skills.py --stage prepare
python scripts/train_two_skills.py --stage smoke
python scripts/train_two_skills.py --stage train --steps 10000
```

train 模式不再执行 smoke，建议优先 all。每次启动都使用独立 UTC 实验名，不自动恢复或重启失败训练。all 中前一个任务出错会停止整轮；已成功任务记录保留，可用 --task tube_pour 或 --task beaker_move 单独继续未完成任务。每次正式训练从基础权重开始，不从 smoke 或另一个任务权重继续。

## 选择模型执行

每次运行生成 `outputs/two_skills/<UTC时间>/run.json`，记录各任务 config、repo_id、receipt、smoke_checkpoint、train_checkpoint 和状态。只有训练进程成功退出且步目录含 params/assets 时才记录检查点；不要拿 smoke_checkpoint 当正式技能模型。

执行程序按 task 名取对应 train_checkpoint，使用同一任务的配置启动已有 serve 入口，例如（在训练节点环境中）：

```bash
PYTHONPATH="$PWD/src" python -m piper_titration \
  --config /run.json中该任务的config路径 \
  serve --checkpoint /run.json中该任务的train_checkpoint路径 --port 8000
```

每次启动前选择所需 GPU（例如 CUDA_VISIBLE_DEVICES=0），保持缓存路径设置。服务会恢复检查点对应的训练配置和归一化资产；切换任务时切换对应服务/检查点，不混用烧杯与倒液的统计。当前 serve 监听节点 localhost，远程客户端需 SSH 转发。此脚本完成训练编排与任务到检查点的记录，不自动连接机械臂或执行动作，也未实现热切换控制器。

本地只验证编排、参数和隔离逻辑，无法据此宣称两个技能已经学会。需远端完成训练，再在固定地点分别执行完整抓放、倒液放回任务确认效果。10000 步是可调整的起始设置，不是效果保证。
