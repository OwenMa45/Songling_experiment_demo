> 服务器已迁移至 4090 + songling Conda。路径、环境创建和运行命令以 [当前服务器说明](server_4090.md) 为准；本文旧 NAS 命令仅留作历史参考。

# 单臂真实示范到 π0.5

当前用户目标为**同一场景的 2+2 技能展示**。固定场景先导训练使用显式 `--train-only`，操作见 [烧杯展示微调流程](beaker_demo_training.md)。下文的三组划分要求适用于默认分组评估模式，不是固定场景展示的训练前置条件。train-only 仍逐条检查全部数据与验证材料，只输出训练集；效果用同场景新执行的完整回合检验。

## 环境

NAS 项目目录为 `/mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo`，原始数据目录为该目录下的 `data/四项任务数据/`，已记录到 `paths.raw_data_root`（可用 `PIPER_RAW_DATA_ROOT` 覆盖）。转换仍使用显式回合选择清单，不会仅凭文件夹名推断成功标签或采集分组。建议在数据根目录下按 `configs/chemical_tasks.json` 的四个 task_id 分目录，再放置各 episode。`data/`、`outputs/` 和权重目录均不进入 Git。

代码推送后由用户在 NAS 执行即可，不需要向本助手提供 SSH 登录信息。详细准备步骤见 [NAS 运行准备](nas_setup.md)。

默认读取 `configs/chemical_server.json`。远程路径和 `PIPER_*` 环境覆盖规则保持不变：仿真 Python 在 `/mnt/cpfs/users/mrq/emboddied/mujoco/bin/python`，训练代码与 `.venv` 在 `/mnt/cpfs/users/mrq/emboddied/openpi` 下。训练配置仍核查 openpi 固定提交，不修改外部代码。

基础依赖为 NumPy、Pillow 和 PyYAML；单臂数据处理不要求安装 MuJoCo。历史仿真依赖移到 `pip install -e '.[simulation]'` 可选组。训练环境使用固定 openpi 自身依赖，不用本项目的安装命令升级其 JAX/Torch/LeRobot。

单臂模型必须使用 `pi05_piper_chemical_lora` 配置；原 `pi05_piper_lora` 为双臂旧实验。两类检查点不得混用。服务器实际 GPU 和基础权重是否就绪仍需核实。

## 选择回合

每回合保持 `metadata.json`、`samples.jsonl` 和 JPEG 图片，使用 `configs/capture_selection.example.json` 的结构指定回合：

```json
{
  "episodes": [
    {"path": "data/episode_0001", "task_id": "beaker_move", "group": "session01_sceneA"}
  ]
}
```

路径相对于选择清单所在目录，支持绝对路径。group 表示实际共享的采集场次/场景；不要为每条连续重复示范随意生成一个新 group。相同 group 即使包含不同任务，也只进入同一数据分区。每个选中的任务至少需要三组，才能形成训练/验证/测试集；如果无法构造各任务均有覆盖的分组，程序会要求补采，而不退回逐帧随机划分。

`capture-plan` 会重新完整检查每个回合，包括图片解码。拒绝 null/非有限/非 7 维动作、未验证来源、非成功标记、损坏图片、不支持的单位等。不接受修改元信息来假装验证已完成；动作与反馈的来源必须先由实际设备和记录程序核实，再如实补充来源审查信息。

当前采集脚本原格式未必包含 `physical_feedback_source_verified`。这属于待核实信息，不能仅通过脚本默认值为 true。该程序仅检查声明和文件一致性，不能替代现场来源验证。

数据集的语言指令由选择清单的 task_id 映射到四任务目录定义，操作者必须确认标签与实际演示相符。

## 时间和图像

只使用记录中的实际 `action`，不会从状态反推命令，也不会在导入阶段猜测 CAN 载荷。机器人采样时间使用 `host_monotonic_ns`，以首帧为起点按配置频率建立均匀网格，在每个网格点选择不晚于该时刻的最近记录。状态、动作和图像始终选择同一原始样本，最大样本年龄不得超过 1.5 个目标周期。出现长缺口时要求分段审查或补录，不无限保持上一帧。

原始记录不修改；审查凭据记录每回合输出帧数、最大重采样年龄、原始 JSONL/元信息哈希。front/wrist 图像按 openpi 的保比例补边方法处理到 224×224。此重采样不修复相机曝光延迟或控制信号来源，采集同步仍须单独验证。

## 输出与训练

`convert-captures` 完成所有回合预检与时间检查后才导入 LeRobot 并创建数据集。默认生成：

- `local/piper_chemical_train`
- `local/piper_chemical_validation`
- `local/piper_chemical_test`

存储位置遵循该 openpi 所锁定 LeRobot 的 `HF_LEROBOT_HOME`。不上传到 Hub。全部导出完成后生成 `paths.dataset_receipt`，默认 `outputs/chemical/dataset_receipt.json`。导出中途失败的目录没有有效凭据，不能用于训练；换一个新的数据集名称和凭据路径重试，不自动删除已有数据。

默认 train repo_id 与转换输出相匹配；若使用自定义 `--repo-id`，同步更新配置中的训练 repo_id。训练前校验该凭据、数据集元信息哈希、路径和 30 Hz 控制约定。数据内容在导出后应保持只读；元信息哈希不等于每个导出图片的完整内容哈希。

训练输入是 7 维单臂状态、两路图像及指令。六关节绝对目标转为相对当前状态的增量，夹爪保持连续绝对开度；模型输出反向转换回 7 维绝对目标。π0.5 内部填充到 32 维，其余维度不映射到第二台机械臂。

默认单卡 LoRA、batch=1、horizon=16。`norms` 仅从 train 集计算统计，validation/test 不参与参数更新。首轮 `train --steps 1` 用于实测权重恢复、反向传播和检查点保存。修改 experiment 后再正式训练，不能把一条重复演示训练成功当成四任务完成。

`serve` 使用训练保存的配置和归一化资产，并声明 `piper-chemical-single-v1`、7 维、30 Hz 接口；旧双臂仿真客户端会拒绝该协议。该服务提供动作预测，不连接真机或发送 CAN 指令。

## 目前还不能完成的验证

已提供回合在数据检查阶段即被拒绝，尚无合格数据集，未运行 LeRobot 实际导出、单步 GPU 训练或正式微调。脚本和单元测试验证的是可执行链路及失败处理，不是已获得训练结果。还需要确认 CAN 物理来源、补齐真实动作/缺图、提供其余回合与可访问的训练环境。
