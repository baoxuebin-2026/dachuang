# 阶段 8 / V1-A2：Bonn 室内人员二维框子实验

记录日期：2026-09-25（北京时间）。这是[项目组已选 A2](stage8-v1a-a2-annotation-kit.md)的**人员二维框**验证。没有三维人体位置真值、危险区越界真值、焦化现场图像、D435 实采或 Jetson 实测。两条 Bonn 序列来自极相似的室内环境；第二条的人工画面和标签已经在质检时查看，不能称为完全盲测。

## 冻结输入与核验

- 原始两包分别从[波恩大学官方来源](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)取得、校验 SHA-256。选定的 115 帧和 24 张复标帧的完整 ID、时间戳、配对深度路径及序列分组由[抽样清单](../data/audits/bonn_v1a_sampling_manifest.csv)确定，其 SHA-256 为 `27f69ecb1e7742223ca82ef95dae3e2aeb5bc067a00dff5683409166e9a9ae32`。
- 两份人工原稿分别完成 115／24 帧，之后依照团队逐帧提交的修正形成两个**明确标作组装版**的非公开输入。[二维评分冻结配置](../configs/stage8_v1a_a2_label_freeze.json)记录两份组装版 SHA-256：主标 `23a27695657334be0bec47dfd3f8ef7e0e3a7ad6e14ef96802ab60945934682e`（115 帧，66 框）；复标 `14ebd71d0c5dc572240895cccf66ab10c1f84d1478f7d79e6d672e85741e964d`（24 帧，14 框）。组装版继承原导出的 `exported_at` 字段，时间**不能**视作组装完成时间。原稿、修订版及最终输入均不放公开 Git。
- 24 帧复标里，14 帧两人各标 1 人，10 帧两人均标无人；14 对框的平均／最低 IoU 为 0.8572／0.7672，属性标志一致。二维评分**预先选用最终主标框**作真值；复标衡量标注差异，不用模型输出反向修改真值。另有 15 个主标框碰图像边界但 `truncated=false`：此标志**未用来筛选评分帧**，完整／贴边两组另列灵敏度。不能据此声称全部属性标志已经逐帧仲裁完毕。

模型在标注前预选为 TorchVision `ssdlite320_mobilenet_v3_large`、`SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`（person 类别 1）；官方权重 SHA-256 为 `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`。配置固定置信度阈值 **0.50** 与一对一匹配 IoU **0.50**，没有根据这两条视频调阈值、训练网络或选取成功帧。[官方说明](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)提供相应权重和推理预处理。PyTorch 2.9.1+cpu、TorchVision 0.24.1+cpu、Python 3.12.14；输入是源 RGB 640×480，由权重自带 transforms 预处理。

## 计算结果

| 统计单位 | 第一条（开发） | 第二条（相似房间复核） | 全部 |
| --- | ---: | ---: | ---: |
| 抽样 RGB 帧 | 58 | 57 | 115 |
| 人工人员框 | 32 | 34 | 66 |
| 模型输出框（≥0.50） | 30 | 32 | 62 |
| TP / FP / FN（IoU≥0.50） | 30 / 0 / 2 | 32 / 0 / 2 | 62 / 0 / 4 |
| Precision | 1.000 | 1.000 | 1.000 |
| Recall | 0.938 | 0.941 | 0.939 |
| F1 | 0.968 | 0.970 | 0.969 |
| 无人帧 / 其中误报帧 | 26 / 0 | 23 / 0 | 49 / 0 |

四个漏检 ID：第一条 `0300`、`0310`；第二条 `0020`、`0330`。其中三帧人工框触及边界，一帧主体较完整。以**预先计算的主标框是否触边**作诊断分层：20 张触边帧 17/20 检出，95 张不触边帧在 46 个人框中检出 45/46；不能把这两组的差异解释成因果、跨场景泛化或现场漏报率。

本次仅在通用电脑 CPU 测量**单图预处理与模型推理**，4 个 PyTorch 线程，3 次开发帧预热；中位延时 **53.298 ms**，最大值以同次原始 `summary.json` 为准。CPU 报告名 `AMD EPYC 9V74 80-Core Processor`；这些数值来自共享运行环境，未测文件读取、人工标签计算、管线排队或 Jetson 设备，不能外推工业现场实时性。

## 复现和边界

运行脚本：[固定权重二维评估](../scripts/run_bonn_a2_detector.py)，软件及权重版本在[预标注配置](../configs/stage8_v1a_a2_preannotation.json)登记。Git 提交 `b070f3996380c2e7b4d48db2c693134aa541305b` 运行，私有输出为 `outputs/stage8_a2_detector/{summary.json,frame_results.csv,person_detections.csv}`；这些逐帧文件单独留存，不公开随原图分发。完整结构核验：

```bash
python scripts/prepare_bonn_v1a.py
python scripts/validate_bonn_v1a_labels.py --primary data/processed/bonn_v1a/labels/bonn_v1a_team_1_primary_rev3_assembled.json --qc data/processed/bonn_v1a/labels/bonn_v1a_team_2_qc_rev2_assembled.json
.venv/bin/python scripts/run_bonn_a2_detector.py
```

上述 `.venv/bin/python` 适用于 Linux；Windows PowerShell 使用 `.venv\Scripts\python.exe scripts\run_bonn_a2_detector.py`。两者均须事先安装与[预标注配置](../configs/stage8_v1a_a2_preannotation.json)一致的 CPU PyTorch/TorchVision，下载并核对官方权重，放在 `data/raw/bonn_rgbd/weights/`；脚本会再次核对两份标注、抽样清单和权重的哈希，不会下载或改写原始文件。

这只回答“公开室内画面的人体框能否作为多源模型的**视觉输入证据**”。原申报书中的 D435 对应未来 RGB-D 感知端、Jetson 对应未来边缘推理端；本文所测并非这两台设备。下一步要单独定义开发序列上的深度有效性与未知状态，提出若干可选择的**相机相对虚拟区域**规则，再由项目组决定是否进行空间接口实验。不能把独立 UCI 气体和 Bonn 人员轨迹当成同期现场记录。
