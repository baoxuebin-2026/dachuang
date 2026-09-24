# 面向山西省焦化园区的智能安全预警方法研究

本仓库记录大学生创新训练计划项目的公开数据来源、可复现实验代码和阶段决策。研究主场景为**管廊气体异常与人员靠近的联合预警**。

## 研究边界

- 气体数据来自公开实验室数据集；人员位置数据如由场景模拟生成，须在文件和图表中标记为“仿真”。
- Intel RealSense D435、NVIDIA Jetson Orin Nano 和现场传感器属于**拟部署系统架构**，项目未使用这些设备采集数据或进行性能测试。
- 实验室气体释放识别、仿真场景风险评估与焦化园区事故预警是不同结论。论文只报告由实验实际支持的结论。
- 申报书中的烟雾遮挡、积液识别及论文发表目标须在后续阶段逐项对照完成情况，不能直接标记为已完成。

## 当前进度

1. 阶段 0：审查申报书，选择“气体动态预警为主、空间风险仿真验证”为研究路线。
2. 阶段 1：确定管廊气体异常与人员靠近为主场景；不使用实体设备。
3. 阶段 2：已确定 **UCI 309 主实验＋UCI 487 独立辅助实验，UCI 322 备用**；参见 [数据方案与决策记录](docs/stage2-data-decision.md)。
4. 阶段 3：已制定 [EDA 拟执行清单](docs/stage3-eda-plan.md)，等待项目组确认检查范围后实施；尚无 EDA 结论。

数据文件不提交到 Git。数据源及归档校验记录见 [data/sources.md](data/sources.md)，下载方法见 [scripts/download_data.py](scripts/download_data.py)，未完成的申报承诺见 [docs/commitments.md](docs/commitments.md)。代码、图表及结论追溯规范见 [本项目研究工作流](docs/research-workflow.md)，上游为 [math-modeling-ai-workflow](https://github.com/baoxuebin-2026/math-modeling-ai-workflow)。

> 本仓库为方法研究与系统拟部署设计，不代表对任何焦化园区完成现场试验或工业级安全认证。
