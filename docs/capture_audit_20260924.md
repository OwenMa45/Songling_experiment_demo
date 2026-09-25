# 2026-09-24 真实遥操数据审查

## 结论

当前提供的 `beaker_move/episode_20260924_182612` **不能直接用于 π0.5 动作监督微调**，但有图像、状态及原始 CAN，可保留用于诊断和经过验证后的片段恢复。没有启动训练，没有生成或宣称存在微调权重。

本次在 `piper_capture_starter/data` 下只发现这一个回合。结论只覆盖本地实际提供的数据，不代表其他未提供的回合。

## 实际检查结果

| 项目 | 结果 |
| --- | --- |
| 帧数、时长 | 703 帧，23.4004 秒 |
| 实际采样率 | 29.9995 Hz |
| 帧间隔 | 最小 30.9685 ms，中位数 33.3374 ms，最大 35.6892 ms |
| 观测状态 | 703 个有限 7 维向量，六关节和夹爪均有变化 |
| 动作标签 | **703/703 为 null** |
| 图片 | 1406 个引用中，1405 张成功解码；缺 `images/front/000060.jpg` |
| 相机重复序号 | front 4 帧，wrist 6 帧 |
| 图像读取相对采样时差 | front 中位数 −14.90 ms；wrist −25.03 ms；绝对最大分别 35.72/34.68 ms |
| 反馈物理来源 | 元信息明确未验证 |
| 回合成功标记 | `task_success=null`，`interrupted_needs_review` |

相机时间是主机读取完成时刻，以上数字不是曝光同步精度。头部画面明显偏暗，随后曝光变亮；建议预热相机、待曝光稳定再开始录制。抽查 front/wrist 的 0、100、200、300、400、500、600、702 帧，能看到抓取、移动和释放烧杯，但这不等同于核实完整操作成功、目标区域或 CAN 信号来源。线缆经过操作区，腕部画面在夹持时存在遮挡，需要在后续采集中优化。

关键元信息不是普通的缺字段：

```json
{
  "format": "piper_x_observation_only_v1",
  "physical_feedback_source_verified": false,
  "data_purpose": "observation_only_diagnostic_not_command_supervision",
  "ready_for_policy_training": false,
  "action_semantics": "not_recorded_always_null_no_imputation"
}
```

不要把这些标记改成 true 或把实测下一帧关节位置写入 action 来通过检查。原始数据保持原样。

## 原始 CAN 是否有挽救可能

`can_raw.log` 中 `0x155/0x156/0x157/0x159` 各有 3583 条消息。这些 ID 与官方 Piper 系列关节目标、夹爪控制报文一致，但**该一致性不能证明 PiPER-X 当前拓扑、固件下的发送者及语义**。来源：[官方主从说明](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/double_piper.MD)、[官方接口说明](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD)。必须再匹配采集引用的 pyAgxArm 提交、PiPER-X 固件和实际 ID 配置。

仅按时间统计、不解码载荷：600/703 帧能够匹配四类完整前序报文，且最旧报文不超过采集配置中的 0.2 秒；这个阈值来自诊断缓存检查，**不是已验证的训练同步容差**。其余帧区间为 0–11、25–31、307、620–702；首条候选目标报文在采样开始约 0.371 秒后，最后一条在约 20.457 秒，而录制持续到 23.400 秒。

可能的恢复步骤：

1. 核实相机和观测状态对应实际执行任务的从臂；确认线束拓扑、CAN ID 偏移、控制报文是否被转发/回显。
2. 对照被引用的官方 SDK 提交验证字节顺序、关节单位、夹爪模式、控制模式及复合关节消息组帧方式。
3. 从真实控制消息生成单独的派生文件，保留原始报文时间、消息 ID、来源与变换记录，不改 `samples.jsonl`。
4. 检查墙钟与单调钟对齐，禁止使用采样之后的报文回填过去动作；长时间无目标更新是否代表“保持”需要由控制器语义证实，不能默认无限向前填充。
5. 找回缺失图像或按明确规则剔除损坏片段，保持时间索引，不把两段不连续轨迹拼成连续动作序列。
6. 复核曝光、抓取过程与成功标签，再判断是否有足够完整的成功片段用于训练。

### 补充核对与候选动作恢复

用户已确认主从共用 `can0` 总线，实际日志使用默认 ID，无 `+0x10/+0x20` 控制 ID。状态随从臂和夹爪动作变化，并已做数值、时间交叉核对；物理发送设备身份仍未最终验证。保留 `physical_feedback_source_verified: false`，待按设备安全流程只接从臂进行只读采样后再记录验证证据。

随后成功取得采集引用的 pyAgxArm 固定提交 `841a625f5f4920e776f20b934eb13048b747e6d0`，核对 PiPER-X v189 的驱动继承关系及解码规则：关节为大端有符号整数、毫度转弧度；夹爪宽度为微米转米。参考：[关节解析器](https://github.com/agilexrobotics/pyAgxArm/blob/841a625f5f4920e776f20b934eb13048b747e6d0/pyAgxArm/protocols/can_protocol/drivers/piper/default/parser.py)、[夹爪解析器](https://github.com/agilexrobotics/pyAgxArm/blob/841a625f5f4920e776f20b934eb13048b747e6d0/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/parser.py)。

新增 `scripts/recover_can_candidates.py`，只生成独立诊断文件，不改原始 action 或元信息。要求按 `155→156→157→159` 顺序组成完整消息组，组内跨度不超过 5 ms；检查已观测控制模式、夹爪使能及零点指令；只匹配采样时刻之前已完整收到的组。5 ms 和 200 ms 都是可配置诊断筛选值，尚不是标定后的训练参数。

本次输出在 `outputs/data_audit/can_recovery_001/`：

- 3583 组完整控制消息；600/703 个采样点有候选动作。
- 连续候选区间（含端点）：12–24、32–306、308–619；12 帧无前序完整组，91 帧目标过旧。
- 缺失 front 图像的第 60 帧仍在候选区间内，因此 600 并非合格训练帧数，也不能把这些区间直接拼接。
- SDK 状态与最近前序 CAN 反馈逐维绝对差值的中位数均为 0；六关节最大差值约 0.00850 rad，夹爪最大差值 0.0005 m。这是异步快照比较，不能证明设备身份或原子同步。
- 观测墙钟减单调钟的跨度为 0.002465 ms；派生文件保留原始文件 SHA-256、报文行号、时间及拒绝原因。

派生字段命名为 `candidate_action`，始终保留 `ready_for_policy_training: false`，不会被训练转换入口当作已合格 action。还需验证设备身份、动作与图像时序语义、缺图处理及任务成功标签。

## 2+2 个任务的采集安排

任务定义存放于 `configs/chemical_tasks.json`：

| 优先级 | 任务 | 计划合格回合 |
| --- | --- | --- |
| 必做 | 抓取烧杯并移动到目标区域 | 200 |
| 必做 | 取试管 → 倒液 → 放回 | 200 |
| 探索 | 滴管取液 → 滴液 | 200 |
| 探索 | 取倒扣滴管 → 反转 → 放入指定孔 | 200 |

这里 200 指**质量合格且任务成功的完整示范回合**；200 条不是保证成功率的理论门槛。失败回合保留并标记，不能混入成功示范。每个任务初始规划 160 训练、20 验证、20 独立测试，按采集场次和初始场景分组，避免相邻重复演示泄漏。统计与调参只能使用训练/验证部分，最终测试保持独立。

建议先完成每个必做任务 5–10 条先导回合，通过动作来源、图像同步和转换检查后再扩展到 200。变化覆盖可达范围内的物体位置、目标位置、容器实例和适度光照，而不是同一场景机械重复 200 次。试管倒液的成功标签须包含真实倒入目标容器并放回，滴管任务须包含实际吸液/出液，不能只凭末端到位判断。

模型输入保持单臂 7 维状态与 front/wrist 图像，通过任务指令区分四类任务；动作应为实际下发的六关节绝对目标与连续夹爪宽度。已增加单臂默认配置、`capture-plan`/`convert-captures` 及单臂训练适配，历史双臂命令需要显式指定旧配置。对当前回合执行单臂预检仍会失败；代码适配完成不代表数据已经合格或训练已经运行。

## 复现本次审查

```bash
python scripts/audit_capture.py piper_capture_starter/data \
  --decode-images --output outputs/data_audit/audit.json
python -m unittest discover -s tests -p test_capture_audit.py -v
python scripts/recover_can_candidates.py \
  piper_capture_starter/data/beaker_move/episode_20260924_182612 \
  --output outputs/data_audit/can_recovery_002
python -m unittest discover -s tests -p test_can_recovery.py -v
```

完整图像解码需要 Pillow；审查不会连接机械臂、补造 action、修改原始数据或发起训练。存在阻断项时返回非零。`outputs/data_audit/contact_sheet.jpg` 是本次生成的视觉抽查图，不作为完整任务成功标注。

拓扑和默认 ID 已得到用户确认；仍需最终设备身份验证、其余回合的位置、训练服务器连接方式和已下载权重的位置。目前尚未运行微调，真机任务成功也未得到验证。
