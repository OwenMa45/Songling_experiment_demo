# 从 NAS 启动 4×3090 节点训练

入口：`scripts/train_remote_3090.py`，仅需 NAS 上 Python 3.9+ 与 OpenSSH 客户端。SSH 登录节点为 10.71.106.251。脚本使用密钥认证，不读取/复制私钥内容，不安装环境，不自动同步数据。

目前仅已确认 IP 与四张 24GB 3090。默认登录用户 lzk、共享 NAS 挂载路径与 4090 节点相同、Conda 为 /home/lzk/miniconda3/bin/conda、环境 songling；这些均为可覆盖假设，须由远端 doctor 检查。共享文件可见不代表 Conda 环境或 GPU 库也已配置。

## 1. 建立 SSH 信任

在发起训练的 NAS 上执行，私钥留在 NAS 本机（不要上传 Git，也无需发给助手）：

```bash
ssh -i /你的私钥路径 lzk@10.71.106.251
```

首次连接核实管理员提供的主机指纹。若私钥有密码，先通过 ssh-agent/ssh-add 解锁。脚本使用 BatchMode=yes 与 StrictHostKeyChecking=yes，遇到未知主机或认证失败会停止，不跳过验证。

## 2. 诊断与单步训练

在 NAS 项目目录运行；替换私钥路径：

```bash
cd /mnt/nas/share/home/lzk/mrq/robot/Songling_experiment_demo
python scripts/train_remote_3090.py doctor --identity /你的私钥路径 --dry-run
python scripts/train_remote_3090.py doctor --identity /你的私钥路径
python scripts/train_remote_3090.py smoke --identity /你的私钥路径
```

默认使用 GPU 0,1,2,3、全局 batch=4、FSDP=4、显存比例 0.9。每次 smoke/train 自动生成独立时间戳实验名，避免覆盖已有失败目录。默认配置 configs/beaker_demo.json，读取已知成功的 outputs/chemical/dataset_receipt.json、data/lerobot，复用 outputs/beaker_demo 中的归一化统计。无需重复转换或计算 norms。

成功完成单步更新和检查点保存后，启动正式训练：

```bash
python scripts/train_remote_3090.py train --identity /你的私钥路径 --steps 10000
```

命令在前台等待并转发输出及退出状态；不会自动后台运行或重启。建议在 NAS 的 tmux 会话中执行，以免用户终端断开中断任务。3090 节点重启或 NAS 到节点 SSH 链路中断不由 tmux 恢复。日志写入远端项目 outputs/remote_logs；检查点仍在 outputs/beaker_demo/checkpoints。不要同时启动相同 GPU 上的多个训练。

## 覆盖节点差异

所有覆盖参数需在 doctor、smoke、train 命令中一致传入：

```bash
python scripts/train_remote_3090.py smoke \
  --identity /你的私钥路径 --user lzk \
  --python /节点上的/songling/bin/python \
  --gpu-list 0,1 --batch-size 2 --fsdp-devices 2
```

也支持 --project、--openpi、--checkpoint、--conda、--env、--config、--receipt、--lerobot、--output、--cache、--hf-cache、--port、--experiment、--memory-fraction。配置/凭据/LeRobot/output 的相对路径以远端项目根目录解析；私钥路径属于发起端 NAS。--python 优先于 --conda/--env。

如果节点以不同路径挂载数据，现有 receipt 的绝对目录校验可能失败，不要改凭据绕过；优先保持同一 NAS 挂载路径，或按新路径重新导出到独立目录。改变 --output 后归一化资产也会换位置，需要在该输出路径运行 norms。

FSDP 使用项目固定 openpi 的实现，上游初始化仍存在复制输入参数的显存峰值；四卡不保证单步一定成功。只有实际日志包含更新、检查点保存并以 0 退出，才能确认跑通。本地验证仅覆盖命令构建、参数约束和路径转义，未连接新节点。
