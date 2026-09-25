# 创建 MuJoCo 环境

在 NAS 项目目录执行（需要 Python 3.11 或 3.12，能访问 Python 包源）：

```bash
cd /mnt/cpfs/users/mrq/emboddied/Songling_experiment_demo
python3.11 scripts/setup_sim_env.py
export PIPER_SIM_PYTHON="$PWD/.venv-sim/bin/python"
export MUJOCO_GL=egl
```

脚本默认创建项目内 `.venv-sim/`，安装本项目 `[simulation]` 依赖、检查依赖冲突，再测试物理步进和无窗口渲染。已有合法 venv 会复用；不会删除环境或修改已有 openpi 训练环境。重复安装可能按依赖约束更新仿真包。

如需沿用此前 NAS 路径：

```bash
python3.11 scripts/setup_sim_env.py --env /mnt/cpfs/users/mrq/emboddied/mujoco
```

此路径必须不存在或是已有 Python 3.11/3.12 venv。若是 Conda 环境或源码目录，脚本会拒绝修改；使用默认 `.venv-sim` 即可。系统需提供 NVIDIA 驱动和可用 EGL 库，脚本不安装系统驱动。暂时只验证物理计算可加 `--skip-render`，但这不代表服务器渲染已就绪。

Windows 本地也可执行：

```powershell
py -3.11 scripts/setup_sim_env.py
$env:PIPER_SIM_PYTHON = "$PWD\.venv-sim\Scripts\python.exe"
```

环境文件 `.venv-sim/` 和项目内 `mujoco/` 已加入 `.gitignore`。仿真场景、模型配置和脚本仍应跟踪到 Git；第三方模型仍通过 submodule 管理。自定义的其他项目内环境目录需要自行加入忽略规则。NAS 项目之外的环境不在本仓库跟踪范围内。

此脚本配置运行依赖并验证最小场景，不生成化学实验场景，也不安装用于策略服务通信的可选 openpi-client。后续需要该客户端时，从已锁定 openpi 的 `packages/openpi-client` 安装到仿真环境。真实数据微调仍使用独立 openpi 训练环境。
