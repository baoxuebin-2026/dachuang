# Bonn 视觉接口的再次运行与证据核对

日期：2026-09-26（北京时间）。这是对[原二维检测](stage8-v1a-a2-2d-detector-results.md)和[原深度／虚拟区域复核](stage8-v1a-a2-repeat-depth-results.md)的**重新运行**，不是新增独立场景、重新调参或重新盲测。

## 输入与方法

- 从波恩大学原发布地址重新下载两段 RGB-D ZIP；其 SHA-256 分别为 `a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5`、`d3ef7898529c60dc39919ea699d00490d98a2c6ae4b165610f2955b235b939b5`，与原[冻结配置](../configs/stage8_v1a_a2_preannotation.json)一致。
- 根据项目组先前在对话中提交的完整主标／质检 JSON 及后续明确修正，恢复两份私有组装版。结构校验：115 帧／66 框、24 帧／14 框。SHA-256 分别为 `23a27695657334be0bec47dfd3f8ef7e0e3a7ad6e14ef96802ab60945934682e`、`14ebd71d0c5dc572240895cccf66ab10c1f84d1478f7d79e6d672e85741e964d`，均与[原标注冻结配置](../configs/stage8_v1a_a2_label_freeze.json)一致；14 对框平均 IoU 0.8572，最低 0.7672。
- 官方 SSDLite320 COCO_V1 权重 SHA-256 `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`；Torch 2.9.1+cpu、TorchVision 0.24.1+cpu，运行原[固定二维脚本](../scripts/run_bonn_a2_detector.py)，置信度 0.50、匹配 IoU 0.50。此次 CPU 为 Intel Xeon Platinum 8573C，**不与旧环境的推理耗时作设备性能比较**。
- 深度复核复用原 A1、G1 冻结配置；[脚本](../scripts/evaluate_bonn_a2_repeat_depth.py)新增加明确的“重生成预测文件”输入和独立输出路径，**原历史默认路径与校验仍保留**。此次不是 2026-09-25 的首次一次性复核，故结果状态另记为 `replication_with_regenerated_detector_predictions`。

## 结果与差异

| 指标 | 原报告 | 本次重跑 |
| --- | ---: | ---: |
| 两段共 115 帧的人工框／检测框 | 66／62 | 66／62 |
| 二维框 TP／FP／FN | 62／0／4 | 62／0／4 |
| 漏检帧 | 第一段 0300、0310；第二段 0020、0330 | 完全相同 |
| 第二段有人的帧／自动框有效深度 | 34／32 | 34／32 |
| 第二段 G1 inside／near／outside／unknown | 2／1／29／2 | 2／1／29／2 |
| 有效双方框的 G1 分类分歧 | 0／30 | 0／30 |

关键差异：重跑的 `person_detections.csv` SHA-256 为 `1f069794ce5bf8fa202e50f32a3d76e1ccae4ab92eaa5951ef04f7612ddf2809`，**不同于旧 A1 冻结配置里记录的 `de91dd98ff42ff8fd4774add5e192393b8c47a7d14fa5e19d13e1b40e2e3806d`**。旧逐帧文件目前不可取得，无法逐数字对比原因；可能与 CPU 数值实现差异有关，但不能据此断言。深度脚本明确校验本次新 SHA、输出到新目录并在结果中同时记录旧 SHA，绝不冒充原运行或改写冻结配置。

公开存档：[本次二维汇总](../results/stage8_bonn_rerun_20260926/detector_summary.json)与[本次深度汇总](../results/stage8_bonn_rerun_20260926/depth_summary.json)。新逐帧 `frame_results.csv`、`person_detections.csv`、`frame_details.json` 和两份汇总另作私有归档；原始 ZIP、人员标注也不进入公开 Git。私有归档 ZIP 的 SHA-256 为 `5c1eb9c146ace5284d968a407cdac61ba37213ad3d096b94918fbc15faec342e`。

**证据边界**：同一两段室内视频、相同人工框与冻结规则的复算只提高结果的可复核性；不能提供独立人体深度真值、焦化现场越界真值、气体和视觉同步观测、D435/Jetson 实机性能或事故提前预警证据。复标是否属于不同实际标注者，应按项目组实际分工表述，不仅凭 JSON 里的 `team_1`／`team_2` 字段推断。
