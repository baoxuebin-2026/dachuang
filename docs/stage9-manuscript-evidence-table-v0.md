# 阶段 9：论文正文证据表 v0（工作稿）

日期：2026-09-26（北京时间）。基于[既定 T1-A 提纲](stage9-t1-manuscript-outline-proposal.md)和[无审阅者时的工作路径](stage9-no-reviewer-progress-and-next-steps.md)整理。**不是新实验或正式论文表格；已从项目组先前消息恢复并哈希核对 Bonn 最终人工标注，仍未重跑模型或复核私有逐帧输出。**正式写作前逐项对应[主张账本](../registries/claim_ledger.csv)。

## 数据来源与评价对象

| 来源 | 观察单位与可用参照 | 本文用途 | 不能证明 |
| --- | --- | --- | --- |
| UCI 309 | 风洞 180 个试验文件，M1 按文件 90/30/60 切分；每次 60 s 开始供气，GC-MS 给的是平均浓度 | 主文气体传感器变化 | 实际事故提前预警、长期阴性表现、逐秒人员暴露浓度 |
| Bonn 两段 `person_tracking` | 相似室内 RGB-D；抽样 58＋57＝115 帧，66 个人工二维框；第二段 34 帧人工有人 | 主文人员框与相机相对空间接口 | D435 精度、禁区越界真值、与 309 同址同步 |
| S1 | 原 309 的 60 个 test 文件与同一套模拟路径及故障反例反复配对 | 主文分级处置规则及质量门控 | 真实联合准确率或独立事故等级真值 |
| UCI 322 | 乙烯＋CO、乙烯＋甲烷各一条气室时序；气体列为**供气设定值** | 独立补充：泛气体响应及干扰 | 目标气种专一识别或真实人受气体影响 |
| UCI 487 | 13 个实验日重复相同 CO 程序，V3 留出 3 日×20 段位 | 独立补充：已完成暴露段 CO 估计 | 实时事故预警 |
| SB112 | 单设备建筑传感器背景日，没有逐秒事故核实 | 独立补充：烟雾传感器背景触发 | 视频烟雾遮挡识别或确定的假事故报警 |

依据：[309 报告](stage5-m1-pilot-results.md)、[Bonn 二维报告](stage8-v1a-a2-2d-detector-results.md)、[深度报告](stage8-v1a-a2-repeat-depth-results.md)、[S1 报告](stage6-s1-fixed-replay-results.md)、[322/SB112 报告](stage7-a-separate-validation-results.md)、[487 报告](stage5-uci487-v3-results.md)。

## 正文主张与结果路径

| 主张 ID | 可写的数值及分母 | 运行结果/图 | 必须同行说明的局限 |
| --- | --- | --- | --- |
| `m1-early-a-multi`、`m1-false-a-multi` | UCI 309 的 60 个留出风洞文件中，57 个在**供气开始之后** 60 s 内提示；释放前每文件 35 s，共 35 min 有 1 个文件曾触发 | [M1 汇总 CSV](../results/m1_uci309/summary.csv)、[运行记录](../results/m1_uci309/run.json)、[登记图](../figures/m1_uci309/m1_test_comparison.svg) | 固定供气时刻；本文件已知洁净起点；评分修订后重用留出；不称事故前预测，程序时钟“60/60”是无效泄漏对照 |
| `stage8-a2-indoor-2d` | 115 帧、66 个人工框，IoU≥0.5 得 TP 62、FP 0、FN 4 | [二维报告](stage8-v1a-a2-2d-detector-results.md)、[私有结果索引](../results/stage8_n1_evidence_index.json) | 第二段曾用于人工质检；最终组装标签已按冻结哈希恢复但不公开，模型逐帧输出尚未重新生成，当前不能独立复算 TP/FP/FN |
| `stage8-a1-repeat-depth-availability`、`stage8-g1-virtual-only` | 第二段 57 帧中 34 帧人工有人，32/34 自动框有有效深度；G1 inside 2、near 1、outside 29、unknown 2 | [深度报告](stage8-v1a-a2-repeat-depth-results.md)、[私有结果索引](../results/stage8_n1_evidence_index.json) | unknown 两例因二维漏检；30/30 相同虚拟区判断只是两框定义的内部一致性；G1 相机横向区不等于 S1 地面区 |
| `stage6-s2-conditional-joint`、`stage6-s4-unknown-zone` | S2 预设模拟重叠 58/60 个已有文件出现 L3；S0、S1、S3 为 0/60；错区、过期和 10 s 缺测各 600 个**重放秒**不可配对 | [场景汇总 CSV](../results/stage6_s1_a/scenario_aggregate.csv)、[运行记录](../results/stage6_s1_a/run.json)、[登记图](../figures/stage6_s1_a/illustrative_s2_timeline.svg) | 同一仿真轨迹复用，L3 来自研究者规则；58/60 不是事故检出率，600=10 s×60 次重放 |

## 补充结果：单独报告

| 项目 | 可写数值 | 来源与限制 |
| --- | --- | --- |
| UCI 322 | B1 对后期时间块的 CO 设定点 14/14、甲烷 13/13 次在 60 s 内响应；事后诊断对仅乙烯变化仍分别 17/17 和 14/14 次响应 | [7A](stage7-a-separate-validation-results.md)、[7B](stage7-b-posthoc-diagnostic-results.md)、[汇总](../results/stage7_a/uci322_summary.csv)；支持泛气体响应，不支持气种专一率 |
| UCI 487 | 验证先选的 14 传感器＋加热相位＋温湿度模型在 60 个留出段位 MAE 0.530 ppm | [运行记录](../results/uci487_v3/run.json)；取每段 15 min 暴露的末 300 s；测试后看到的另一模型 0.463 ppm 不回写为主结果 |
| SB112 | 烟雾通道在 2022-03-10 的 23.978 h 覆盖内去抖合并为 6 个背景触发事件 | [汇总](../results/stage7_a/sb112_summary.csv)；无逐秒环境事故真值，不能称六次经核实的假事故 |

## 投稿前仍需补的核对

1. **人工框 JSON 已从项目组先前消息恢复，哈希与冻结记录吻合，无需重传。**两段 Bonn 原始 ZIP、`outputs/stage8_a2_detector` 及 `outputs/stage8_a2_repeat_depth` 尚不在当前工作区；先按[现有哈希索引](../results/stage8_n1_evidence_index.json)寻找，找不到则从原发布方重新取得 ZIP、重跑固定脚本并新记运行版本，不伪称既有私有输出仍在。
2. 原始研究全文、背景事实及阈值依据逐条核实；1 m 是仿真缓冲假设，不是法定距离。
3. 用[图表登记](../registries/figure_evidence.csv)核图注、图像尺寸和单位；气体实验、室内视觉与仿真规则各自成表成图。
4. 与导师核对原申报尚缺功能的中期/结题表述和论文资助编号；完整论文尚未写完、投稿或发表。

**下一步：**以本表为来源，先起草“研究问题及数据与方法”章节供项目组审查，再写实验结果；不得预造现场效果。
