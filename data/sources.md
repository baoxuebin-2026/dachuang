# 数据来源候选表（阶段 2，待项目组确定使用方案）

初核日期：2026-09-24；E1 增补核查：2026-09-25；UCI 362 候选核查：2026-09-26。UCI 309、322、487、362 的压缩包已实际下载、完成 ZIP 完整性检查并记录 SHA-256；TADI 与 SB112 的公开原文件现已下载并核对发布的 MD5，结果见 [E1 源核查](../docs/stage7-e1-external-data-audit.md)；未注明者仍仅核查公开页面。本表不是焦化园区现场数据清单。

| 数据集与原始下载入口 | 规模、主要字段与时间信息 | 浓度／事故标签 | 与本项目关系及限制 | 许可与结论 |
| --- | --- | --- | --- | --- |
| [UCI 309：湍流混合气体](https://archive.ics.uci.edu/dataset/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.) · [ZIP](https://archive.ics.uci.edu/static/public/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip) | 180 次独立风洞实验，每次约 300 s；原始 50 Hz，另有 10 Hz 版本；时间、温度、湿度、8 路气体传感器；一组一个文件。 | 配置的乙烯＋甲烷或乙烯＋CO 释放档位，释放始于约 60 s、持续约 180 s；气体浓度仅有各档位**实验平均估计值**，无逐时刻真实浓度或事故等级标签。 | **优先主实验**：检验释放识别、背景误报、检测延迟和抗扰；风洞不是焦化现场。训练／测试须按完整实验分组，不可在同一组序列中随机拆窗口。 | UCI 标注 CC BY 4.0；推荐主实验。 |
| [UCI 322：动态混合气体](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures) · [ZIP](https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip) | 官方页显示 4,178,504 行，**E1 实际文件计数**为 CO 4,208,261 行＋甲烷 4,178,504 行，共 8,386,765 行；2 条约 11.6 h 实验（乙烯＋甲烷、乙烯＋CO）；100 Hz 标称；时间、两种气体浓度设定值、16 路传感器响应。 | 有**设定浓度**，不是独立焦化现场检测值；无事故等级。甲烷与 CO 不在同一条实验序列。 | 可用于浓度变化与长序列响应的**独立辅助实验**；仅两条实验序列，样本行数不等于独立实验数；首段固定方案以及仅用两位小数记录时间造成的重复时间戳见 E1 审计。 | 页面许可栏 CC BY 4.0，但正文另写只限科研、排除商用；本项目科研使用，暂不分发原包。 |
| [UCI 487：CO＋温湿度](https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation) · [ZIP](https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip) | 4,095,000 行、13 天实验；时间、CO、湿度、温度、流量、加热电压、14 路传感器；气体传感器约 3.5 Hz，温湿度参考每 5 s。 | 实验室产生并记录的 CO 浓度；无事故等级。 | **推荐独立辅助实验**：检验湿度变化下的 CO 感知；设备、工况与 309 不同，不构成同步多源数据。 | CC BY 4.0。 |
| [UCI 362：家庭活动气味监测](https://archive.ics.uci.edu/dataset/362/gas+sensors+for+home+activity+monitoring) | 元数据 100 条（酒 36、香蕉 33、背景 31）；**实际原文件只有 99 个有时序的 ID**，缺背景 `id=95`；逐行脚本计得 928,991 行，与官网 919,438 实例不符，差异原因待查；约 1 Hz，`id,time,R1–R8,Temp.,Humidity`，元数据另有日期、类别、起点和刺激时长。 | 人为酒／香蕉刺激起点和类别；背景记录未人为刺激；**无 CO／CH4 浓度、事故/危险等级标签**。 | 候选的**异设备变时长刺激及长背景压力测试**，用于有限检验泛气味响应和门控；不得解释为焦化危险气体外测。[源审计与冻结协议](../docs/stage9-a1-uci362-source-audit-and-protocol.md)。 | CC BY 4.0；原 ZIP 已下载并核对外/内层 CRC 和 SHA-256；尚未运行模型。 |
| [UCI 1081：低浓度气体](https://archive.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration) | 90 个独立样本；六种 VOC，各 50／100／200 ppb；10 路传感器，每样本拼接 9000 点，1 Hz。 | 有气体类型与浓度档位；无事故标签。 | 不含甲烷／CO，主要验证微弱 VOC 检测，与主场景对应弱。 | CC BY 4.0；暂不使用。 |
| [Tennessee Eastman Process 仿真数据与代码](https://github.com/camaramm/tennessee-eastman-profBraatz) | 正常与 21 类过程故障；示例训练文件各 480×52、测试文件各 960×52，变量含过程测量与控制量。 | 有**仿真过程故障类型**，不是人员暴露或园区泄漏事故标签。 | 可作为化工过程故障方法对照，但与主场景迁移跨度大。 | 原仓库附许可与署名条件；暂不使用。 |
| [ISR RGB-D Dataset](https://github.com/hcmr-lab/ISR_RGB-D_Dataset) | 项目页面描述 10,000 帧 D435 实验室 RGB-D 序列，含 person 框标注。 | 人员类别标注；没有焦化危险区域／环境气体联合标签。 | 仅作为未来视觉模块**独立验证候选**；数据下载及再使用许可待核验。 | 暂不下载，不能写作本项目 D435 实测。 |
| [TADI-2019 甲烷受控释放](https://zenodo.org/records/8399829) | 6 个记录器共 38,826 行、CO2/CH4 受控释放实验中的 CH4 参考实测及多路 MOX 响应、部分温湿压、编号；行间隔主要 6/12 s，记录只覆盖已编号释放片段。 | 有参考仪器 CH4 浓度及释放编号，**无逐行阴性／精确释放起点字段**。 | 有受控现场多设备差异；没有现场长时无释放对照，缺失哨兵 `-9999.0`；不同于焦化园区。 | Zenodo API 记录 CC BY 4.0；9 个文件 MD5 已验证；条件性补充。 |
| [SB112 建筑消防与气体记录](https://zenodo.org/records/6616632) | 4 个传感器文件、各含 8–11 日工作表；时间、类型、单位、读数，约 20 s；本项目已下载 CO/CNG/LPG/烟雾共 35,649 点。 | 作者注明 8–10 日无物理触发；11 日有近距离刺激，但**无精确开始秒标签**。 | 独立仪器内部几天阴性对照；不是焦化工况，不能未经标定套用 309 阈值。 | Zenodo API 记录 CC BY 4.0；4 个文件 MD5 已验证；仅候选。 |
| [Bonn RGB-D Dynamic Dataset：两条人员跟踪序列](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) · [ZIP 1](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking.zip) · [ZIP 2](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking2.zip) | **两个原包都已下载并核验**：第一条 580 对、第二条 567 对 640×480 RGB／16-bit 深度 PNG，各约 19 秒；相机内参及相机轨迹。`depth.txt` 分别多 2／3 条无文件记录，需按实际文件和时间戳配对。 | 无人工人员框、人员三维真值或危险区标签；相机轨迹真值不可当人员位置真值。 | 阶段 V1-A 视觉来源；可人工标 2D 人员框，计算深度支持和虚拟区域规则响应；两段画面为非常相似的同一环境，留出另一段也**不代表跨场景泛化**。详见[第一条归档核验](../docs/stage8-v1a-bonn-data-audit.md)和[第二条核验及协议](../docs/stage8-v1a-two-sequence-protocol-proposal.md)。 | 官网要求引用论文，未找到明确再分发许可；原包不进公开 Git；正式评估协议待项目组决策。 |

**数据口径：**UCI 309、322、487 都是实验室采集的真实传感器数据；TADI 是受控场地实验，SB112 是楼宇传感器实验，TEP 属过程仿真；人员危险区轨迹如由本项目生成，必须在每个实验和图表中标记“仿真”。不同来源的时间戳不可假定相同，不能凭同步表格拼接出“真实多源焦化园区记录”。风险等级须单独建立依据，不能由模型自身生成标签再据此报告准确率。

## 已验证归档

| UCI 编号 | 原始 ZIP 大小（字节） | SHA-256 | 文件结构及校验 |
| --- | ---: | --- | --- |
| 309 | 23,403,793 | `5e9b707e7a44b3dcaf39b62bb46597403fcdd98ec89814b15e4876694db5f41e` | 180 个 raw 文件＋180 个 downsampled 文件；ZIP CRC 完整 |
| 322 | 369,001,314 | `8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7` | `ethylene_CO.txt` 和 `ethylene_methane.txt`；ZIP CRC 完整 |
| 487 | 183,298,753 | `cb5ce4a6af1a51b933d1979952d7845f0e9baac54e15a81a1e3599e8b85905d4` | 外层含 README 与嵌套 ZIP；13 个 CSV 在嵌套 ZIP 中；内外层 ZIP CRC 完整 |
| 362 | 29,055,749 | `7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395` | 外层元数据和内层时序 ZIP；内外层 CRC 完整；元数据 `id=95` 在时序中缺失；原包行数与官网摘要不符 |
| Bonn `person_tracking` | 329,482,910 | `a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5` | 580 张 RGB＋580 张深度；CRC 完整；下载自原作者站点；[脚本与 JSON](audits/bonn_person_tracking.json)可复核 |
| Bonn `person_tracking2` | 324,262,783 | `d3ef7898529c60dc39919ea699d00490d98a2c6ae4b165610f2955b235b939b5` | 567 张 RGB＋567 张深度；CRC 完整；下载自原作者站点；[脚本与 JSON](audits/bonn_person_tracking2.json)可复核 |

下载与校验：`python scripts/download_data.py 309`；其余编号同理。原始数据始终保存在 `data/raw/`，不提交 Git。
