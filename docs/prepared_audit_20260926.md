# 2026-09-26 预处理轨迹复核

对象：`piper_capture_starter/data/prepared_data/beaker_move_demo_site_A/episodes/episode_20260924_182612`。

结论：动作数值和轨迹结构已具备训练数据的基本条件，但目前不能通过本项目训练导入。不能仅凭 metadata 的 ready=true 放行。未运行 LeRobot 导出或 GPU 微调。

## 已核实

- 647 帧，约 30 Hz；保留原始索引 12–658，索引连续，无中间删帧拼接。
- 647 个有限七维动作均与 provenance 引用的原始 CAN 行解码数值相符，且引用报文不晚于样本时间。状态、墙钟和单调钟均与对应原始行一致。
- metadata 内四个原始文件 SHA-256 均匹配本地原件。
- 图像清单包含 1294 个条目；初次检查缺 3 张，其余文件哈希均与清单一致。
- 从原始目录恢复了以下三张图片，复制前已验证原件 SHA-256 与清单一致，没有插值或生成图片：

| 预处理路径 | 原始路径 |
| --- | --- |
| images/wrist/000162.jpg | images/wrist/000174.jpg |
| images/wrist/000231.jpg | images/wrist/000243.jpg |
| images/front/000637.jpg | images/front/000649.jpg |

恢复后重新完成图像解码，缺图阻断消除。原始回合、预处理 samples 和 metadata 均未修改。

## 剩余问题

1. metadata 引用的 `../../verification/follower_identity_v1.json` 和 `../../verification/hold_behavior_verification_v2.json` 不存在。需把整个数据包的 verification 目录一并提供并保持相对路径，而不只传 episodes。记录必须与 metadata 声明的 SHA-256 匹配。此处只能判断证据未附，不能断言现场验证未做。
2. 582 帧使用近期目标，22 帧使用内部暂停保持，43 帧使用末尾保持；最大目标年龄分别约 402 ms（内部）和 1478 ms（末尾）。数值能对应旧 CAN 目标不等于证明控制器一直保持，因此需要上述 hold 行为记录支持。
3. human_review 的 success_source 指向 manual_review.csv，但本地包没有该文件；visual_source 写的是 explicit_assume_visual_ok_flag_plus_automatic_decode_checks。这表示视觉质量标志含默认假设，不能称为逐回合人工视觉复核证据。请附上实际审核结果，确认释放烧杯及最终稳定状态在裁剪范围内。
4. 当前导入器接受的动作语义字符串以 absolute_leader_transmitted_joint_targets 开头，新格式使用 absolute recorded CAN target...，因此预检返回 command_semantics_not_verified。这是导入适配问题，不代表已发现动作数值错误；应在核实完整 v2 数据契约和证据后增加支持，不能仅改字符串绕过检查。
5. 172 帧夹爪目标为负，最小 -0.0026 m。解码确认来自原始有符号 CAN 值，并非预处理错误；部署和采集端需统一为同一有符号坐标约定。若改为物理开度，应标定后同时转换状态、动作及运行时接口，不能只截断训练动作。

当前单回合只能用于样本检查，完整数据集仍需按真实采集场次/场景划分训练、验证、测试。不要复制此回合构造三份分区。原始 CAN 或其可访问的归档也应保留以便追溯。

## 本地报告

- `outputs/data_audit/prepared_20260926.json`：修复前图像/结构检查。
- `outputs/data_audit/prepared_deep_check.json`：动作、源文件、图片清单和证据路径交叉核对。
- `outputs/data_audit/prepared_20260926_after_restore.json`：补回原图后的结构检查。

现有通用审查器只检查来源声明，不会自动验证上述外部证据文件；其 blockers 并不覆盖本次人工结合 provenance 发现的全部事项。
