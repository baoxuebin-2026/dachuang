# 阶段 8 / V1-A：替代 LIRIS 的公开 RGB-D 数据实物核验

核验日期：2026-09-25（北京时间）。项目组要求另找数据。本次选择[波恩大学 Bonn RGB-D Dynamic Dataset 官方页面](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)的 `rgbd_bonn_person_tracking`，**从发布方原始 ZIP 下载并实际检查**。本报告取代[原协议](stage8-v1a-visual-source-gate-and-protocol.md)中“优先取得 LIRIS D1”的**来源优先级**；实验目标和“先审数据、后确认协议、再做视觉实验”的顺序不变。

## 已取得数据和核验结果

| 检查项 | 结果 |
| --- | --- |
| 官方直链 | [rgbd_bonn_person_tracking.zip](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking.zip)，HTTP 200、`application/zip`；支持 HTTP Range。 |
| 保存位置与分发 | `data/raw/bonn_rgbd/rgbd_bonn_person_tracking.zip`；本地原包，不进入 Git。 |
| 原 ZIP 大小／SHA-256 | **329,482,910 字节**；`a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5`。ZIP 所有条目 CRC 检查通过。此哈希是**本项目计算的核验值**，未见发布方提供的独立哈希。 |
| 图像 | 580 张 RGB PNG（640×480、8 bit）与 580 张深度 PNG（640×480、16 bit、单通道）；抽样解码三处画面及深度，画面有人员、边缘裁切与无人状态。 |
| 时间与相机姿态 | `rgb.txt` 580 行，`depth.txt` 582 行，`groundtruth.txt` 583 行；RGB 记录约 19.404 秒。逐行清单**不能直接配对**：`depth.txt` 有 2 个路径不在 ZIP 中：`depth/1548265888.07778.png`、`depth/1548265895.68576.png`。仅用实际存在文件按最近时间戳配对，580 张 RGB 对应 580 张**各不重复**的深度图；时间差中位 **7.235 ms**、最大 **16.660 ms**。 |
| 抽查深度 | 第 0／290／579 张深度 PNG 有效（非零）像素占比约 **86.12%／84.65%／85.49%**；三个样本的非零中位原始值约 13835／11430／12515。波恩官网说明数据采用 TUM 格式且深度已与 RGB 配准；[TUM 格式说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)给出 16-bit PNG 的标度 5000 单位／米、0 表示无效。按该标度，三样本非零中位约 2.767／2.286／2.503 m。数值是**图像像素的抽样统计**，不是人员的测距真值或准确率。 |
| 相机标定 | [官网](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)给出 RGB 内参 `fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098` 与畸变参数；称深度图已对齐 RGB。归档包含**相机**运动捕捉轨迹，**没有人员三维真值**。 |
| 人员标签与许可 | ZIP 中有 RGB、深度、时间及相机位姿，**没有人工人员检测框或危险区标签**；官方页面要求引用 ReFusion 论文，未发现明确的数据再分发许可证。保留原包于本地，不将原图推送公开仓库；正式论文展示图例前再核对发布方的使用条件。 |

可复核的结构、哈希、CRC 和配对统计：`python scripts/audit_bonn_rgbd.py data/raw/bonn_rgbd/rgbd_bonn_person_tracking.zip`；原始审查记录在 [JSON](../data/audits/bonn_person_tracking.json)。抽样深度的像素统计在本次实际解码时完成，仅用于判断字段是否合理，**不作为正式人员距离评价**。

## 这份数据能解决什么

与 LIRIS D1 当前下载页超时相比，本数据**原始文件确实取得**，有真实连续 RGB、16-bit 深度、相机内参和可复核的时间戳。适合 [V1-A 协议](stage8-v1a-visual-source-gate-and-protocol.md)中的“小样本人工 2D 人员框—深度投影—虚拟区域规则”接口；无须把动作框改叫人员框，也不必为了视觉子实验训练 YOLO。

主要代价：这一条只有约 19 秒、同一实验房间，且没有人员检测真值；可见人员的独立 2D 框要人工标注。**单条序列的密集相邻帧不是 580 个独立场景**；若要评估检测泛化，需要预先另取[官方第二条人员跟踪序列](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking2.zip)或其他独立会话并按完整序列留出。官网给第二条约 324.3 MB，直链可被网页抓取，但**本项目还没有下载或核验第二条**。

原任务是动态 SLAM：设备的运动捕捉真值测量的是**相机位姿**。如果规则要表示固定在“园区地面”的危险区，需要相机轨迹、世界坐标变换和研究者设定的区域锚点，并清楚标注为模拟；若只使用随相机移动的虚拟体积，则只能称“相机相对区域”，不可写作固定管廊越界。没有独立人员位置真值时，不报告三维位置 RMSE 或真实越界 F1。本序列来自实验室房间，**不是 D435、焦化厂或与气体数据同步采集**。

## 下一阶段建议及决策节点

**推荐**用 Bonn 取代 LIRIS 作为 V1-A 数据源；先取得第二条独立人员跟踪序列并重复本报告的完整性核验，再提交两条序列的帧抽样、人工 2D 标注和留出方案。若第二条无法取得、无明确使用条件或两条过于相似，只做数据格式与局部规则演示，不声称跨场景泛化。此前不运行正式人员检测或空间预警计分。

### 【需要人类决策】

是否选用**已实际下载核验的 Bonn RGB-D Dynamic Dataset** 替换 LIRIS，并允许按上述顺序继续核验第二条序列、提交正式标注／评估协议？推荐选用；无需项目组自行下载第一条归档。
