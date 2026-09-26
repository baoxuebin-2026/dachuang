# 面向山西省焦化园区的智能安全预警方法研究

本仓库记录大学生创新训练计划项目的公开数据来源、可复现实验代码和阶段决策。研究主场景为**管廊气体异常与人员靠近的联合预警**。

## 研究边界

- 气体数据来自公开实验室与家庭气味刺激数据集；E1 另核查了受控场地与智慧楼宇来源，其中 SB112 已用于独立背景触发检查，TADI 尚未用于建模；人员位置如由场景模拟生成，须在文件和图表中标记为“仿真”。
- Intel RealSense D435、NVIDIA Jetson Orin Nano 和现场传感器属于**拟部署系统架构**，项目未使用这些设备采集数据或进行性能测试。
- 实验室气体释放识别、仿真场景风险评估与焦化园区事故预警是不同结论。论文只报告由实验实际支持的结论。
- 申报书中的烟雾遮挡、积液识别及论文发表目标须在后续阶段逐项对照完成情况，不能直接标记为已完成。

## 当前进度

1. 阶段 0：审查申报书，选择“气体动态预警为主、空间风险仿真验证”为研究路线。
2. 阶段 1：确定管廊气体异常与人员靠近为主场景；不使用实体设备。
3. 阶段 2：已确定 **UCI 309 主实验＋UCI 487 独立辅助实验，UCI 322 备用**；参见 [数据方案与决策记录](docs/stage2-data-decision.md)。
4. 阶段 3：完成 309／487 的 [EDA 报告](docs/stage3-eda-report.md)、[可复现脚本](scripts/eda_stage3.py)、诊断图与每实验／每日清单；项目组确认“实验事实与仿真风险分层”。
5. 阶段 4：项目组确定 [可解释仿真处置优先级方案 S1](docs/stage4-risk-definition-proposal.md)，确认 [四状态矩阵、1 m 仿真缓冲及 3 s 气体软证据的敏感性试验起点](docs/stage4-s1-grading-options.md)；这些数值不是现场安全标准或设备实测参数。
6. 阶段 5：已批准 [M1 实验协议](docs/stage5-m1-experiment-protocol-proposal.md)（A 清洁起点主实验、B 统一基线敏感性），完成 [UCI 309 文件级气体变化实验](docs/stage5-m1-pilot-results.md)与[可复现计算程序](scripts/run_m1_uci309.py)。在 60 个留出风洞实验中，A 下八通道中位汇总有 57 个于释放后 60 s 内触发、1 个于释放前触发；**这不是焦化厂事故预警效果**。固定释放时钟仍可伪造理想成绩。
7. 阶段 5 独立辅助研究：项目组已确认 V1＋V3，完成 UCI 487 [双轴阻断下的实验室 CO 校准](docs/stage5-uci487-v3-results.md)、[计算脚本](scripts/run_uci487_v3.py)与图表。13 日各 100 个 CO 暴露段的**段位顺序完全重复**，故同时按日期与段位留出；验证选出的相位＋温湿度模型在 3 天×20 未见段位上 MAE 为 0.530 ppm。这是**完成暴露段后的实验室估计**，不是现场实时预警。
8. 阶段 6：已选定[固定轨迹方案 A](docs/stage6-s1-scenario-design-options.md)，先[冻结配置](configs/stage6_s1_fixed_a.json)，再用既有 60 个 UCI 309 留出实验做 [S1 模拟人员场景重放](docs/stage6-s1-fixed-replay-results.md)。固定“可能重叠”的情境中 58／60 条实验出现 L3；这完全取决于人为轨迹与研究内分级规则，**不属于事故预警准确率或真实多源同步验证**。
9. 阶段 E1：项目组选择调查新的外部证据；已下载核查 UCI 322、TADI、SB112 的[数据源候选报告](docs/stage7-e1-external-data-audit.md)，未运行新模型。此后项目组选择方案 A。
10. 阶段 7A：项目组选择 E1 方案 A 并批准[分离验证协议](docs/stage7-a-two-source-experiment-protocol-proposal.md)，先[冻结参数](configs/stage7_a_protocol_candidate.json)，再用[程序](scripts/run_stage7_a.py)分别完成 UCI 322、SB112 [实验结果](docs/stage7-a-separate-validation-results.md)。CO 气室未来时段多通道检出 14／14，单通道 0／14；SB112 烟雾通道在未人工触发测试日有 6 次／约 24 h 背景报警；**两者都不是焦化厂现场测试**。
11. 阶段 7B：项目组选择 A 并追问单通道漏检；先[固定事后诊断范围](configs/stage7_b_posthoc_plan.json)，后进行[事后诊断](docs/stage7-b-posthoc-diagnostic-results.md)。CO 单通道 14 个测试起点的 60 秒峰值均未达训练阈值；B1 对仅乙烯设定变化也全部报警，故不可称为 CO／甲烷气种专一检测；7B 不修改 7A 原结果。
12. 阶段 8：已取得波恩大学[两条相似室内人员序列](docs/stage8-v1a-two-sequence-protocol-proposal.md)，完成[115 帧人工框及固定模型的二维检测](docs/stage8-v1a-a2-2d-detector-results.md)、[第二条序列深度门控与虚拟区复核](docs/stage8-v1a-a2-repeat-depth-results.md)及[拟部署系统接口与申报目标对照](docs/stage8-n1-system-evidence-map.md)。没有独立人体距离或真实禁区真值。
13. 阶段 9：已确定[论文 T1-A 主次](docs/stage9-t1-manuscript-outline-proposal.md)、完成[证据缺口核查](docs/stage9-evidence-literature-gap-audit.md)及[正文候选证据表 v0](docs/stage9-manuscript-evidence-table-v0.md)。[独立审阅挑战 B](docs/stage9-b-independent-review-challenge-protocol-proposal.md)因目前无独立审阅者暂缓；[UCI 362 日期分组后的一次实验](docs/stage9-a1-uci362-locked-results.md)八通道测试 0/13、R1 1/13，表明该冻结规则在家庭酒/香蕉刺激中不能证明有效跨环境检出，且非焦化气体现场数据。按[进度与下一步](docs/stage9-no-reviewer-progress-and-next-steps.md)继续阶段决策。论文未写完或投稿。

数据文件不提交到 Git。数据源及归档校验记录见 [data/sources.md](data/sources.md)；UCI 下载方法见 [scripts/download_data.py](scripts/download_data.py)。**Bonn 视觉序列**运行 `python scripts/prepare_bonn_v1a.py`，缺少原包时脚本会从官网自动下载并校验；Windows 用 `py -3 scripts\prepare_bonn_v1a.py`，详见[标注操作说明](docs/stage8-v1a-a2-annotation-kit.md)。未完成的申报承诺见 [docs/commitments.md](docs/commitments.md)。代码、图表及结论追溯规范见 [本项目研究工作流](docs/research-workflow.md)，上游为 [math-modeling-ai-workflow](https://github.com/baoxuebin-2026/math-modeling-ai-workflow)。

> 本仓库为方法研究与系统拟部署设计，不代表对任何焦化园区完成现场试验或工业级安全认证。
