# 阶段 8 / V1-A2：已准备的抽样与独立人员框标注工具

记录日期：2026-09-25（北京时间）。项目组从[A2／A1 方案](stage8-v1a-two-sequence-protocol-proposal.md)中选择 **A2**。本阶段**已完成可复核数据准备**，尚无独立人工人员框，尚未运行人员检测器、计算成绩或生成真实越界标签。

## 1. 已冻结的可执行部分

- 从 Bonn 官网上[第一条](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking.zip)与[第二条](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_person_tracking2.zip)原始 ZIP，**各按 RGB 清单的第 0、10、20… 帧**取样；实际得到第一条 58 帧，第二条 57 帧，共 115 帧，按整条序列分为开发组／相似房间留出组。RGB／深度仅按实际 ZIP 文件、最近时间戳和一对一关系配对，最大允许 17 ms。具体记录见[完整抽帧清单 CSV](../data/audits/bonn_v1a_sampling_manifest.csv)，SHA-256 为 `27f69ecb1e7742223ca82ef95dae3e2aeb5bc067a00dff5683409166e9a9ae32`。
- 第二名标注者只标每条序列索引 0、50、100… 的 12 帧，共 24 帧；这 24 帧同时由第一名标注者标。第二名标注者标注期间不得查看第一人的框、检测器输出或评估分数。
- [准备脚本](../scripts/prepare_bonn_v1a.py)在缺少原包时自动从波恩大学官网下载并断点续传，校验两个原包 SHA，生成固定 CSV 及本地 115 张 RGB 画面；画面位于被 Git 忽略的 `data/processed/bonn_v1a/frames/`。原始压缩包与抽样图片都不在公开仓库中。[本机标注网页](../tools/annotate_bonn_v1a.html)会复制进工作区并提供拖框、显式确认无人、遮挡／截断／不确定标记、自动暂存和 JSON 导出。[完整性校验脚本](../scripts/validate_bonn_v1a_labels.py)要求两名标注员各交齐自己的 115／24 帧，并逐项检查坐标、状态和类别属性；结构检查不代表人工标注本身正确。
- 无人帧**明确点“确认无人”**，不要留空；人物只框画面中实际可见的身体部分，不猜被遮挡位置，出画、遮挡另打标记。框是原 RGB 图像的 `(x1,y1,x2,y2)` 像素坐标；机器人、海报上的人物不算真人。两位标注者分别导出完整 JSON；分歧保留双方原稿，再由团队记录仲裁，不以检测器框充当裁判。

本项目工作区已生成并校验了 115 张原图、115 行清单及 HTML／JSON 文件的本机可访问性。由于这些序列没有明确再分发许可，**本项目不向公开 Git 推送原图或原始 ZIP**；项目成员在自己的电脑复现时，更新仓库后直接运行准备脚本，缺少的两个 ZIP 会自动下载到 `data/raw/bonn_rgbd/`（合计约 654 MB）。若团队已下载 ZIP，也可按相同文件名手动放入该目录；程序会校验大小和 SHA，不会覆盖现有坏文件。当前助理执行环境的原包留在本地，不保证长期在线；CSV 与工具保存在 Git。

## 2. 两人标注的直接操作

在仓库根目录、安装 Python 3.12 的环境执行（这一步只使用标准库）：

```bash
python scripts/prepare_bonn_v1a.py
python -m http.server 8765 --bind 127.0.0.1 --directory data/processed/bonn_v1a
```

**Windows PowerShell**（同样在仓库根目录）：

```powershell
py -3 scripts\prepare_bonn_v1a.py
py -3 -m http.server 8765 --bind 127.0.0.1 --directory data\processed\bonn_v1a
```

如果官网在你们的网络无法访问，脚本会保留 `.zip.part` 文件，重运行可续传；也可自行从上面的官方链接下载两个 ZIP，放在 `data\raw\bonn_rgbd\`，名字分别为 `rgbd_bonn_person_tracking.zip` 和 `rgbd_bonn_person_tracking2.zip`。在离线环境用 `py -3 scripts\prepare_bonn_v1a.py --offline` 可获得明确的缺失文件及官方地址提示。脚本默认的原包与输出路径以**脚本所在仓库**定位，不依赖当前命令行目录；但启动 `http.server` 和下面的标注校验命令仍建议在仓库根目录运行。

浏览器打开 `http://127.0.0.1:8765/annotate.html`。第一名标注者填自己的编号，选“全量 115 帧”；第二名使用另一编号，选“独立复标 24 帧”。不要在一个页面更改身份；另一个人用自己的浏览器或电脑打开页面，不共享浏览器本地存储。进度只保存在当前浏览器，务必定期点击“导出我的 JSON”备份。两份 JSON 属原始标注记录；统一保存到 `data/processed/bonn_v1a/labels/`（Git 忽略），之后运行：

```bash
python scripts/validate_bonn_v1a_labels.py \
  --primary data/processed/bonn_v1a/labels/bonn_v1a_team_member_1_primary.json \
  --qc data/processed/bonn_v1a/labels/bonn_v1a_team_member_2_qc.json
```

文件名用实际导出的名字替换。校验成功后才汇总 24 帧的标注差异、仲裁，并冻结最终框真值。若团队暂时只有一人，不要把同一人的两次提交称为“双人独立复标”；可先保留单人 115 帧为**初标数据**，待第二人补齐复核后再做正式成绩。

## 3. 检测器预选与当前执行边界

在看标注前预选 TorchVision 的 `ssdlite320_mobilenet_v3_large`、显式 `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1` 和 `person` 类别。官方文档记录约 3.44 M 参数／0.58 GFLOPS、COCO 人员类别及官方预处理；固定候选阈值为置信分数 **0.50**、一对一匹配 IoU **0.50**，并限定按完整序列汇总。来源、参数及禁止推论记录在[预标注配置](../configs/stage8_v1a_a2_preannotation.json)。官方[权重地址](https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth)已实际下载核验：14,069,355 字节，SHA-256 `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`；权重也不推送公开仓库。

官方代码仓库是 BSD-3-Clause，但说明**预训练权重可能受训练数据的单独条件约束**；所以目前只在非公开科研环境使用并保留来源，不把代码许可证误写成权重的单独授权。当前环境没有安装 PyTorch／TorchVision，权重**未加载或推理**；需要在人工标注完成、运行环境固定且脚本审查后再执行正式检测。暂不声称 PC 或 Jetson 速度。

深度点有效比例、中央身体候选区域与相机相对虚拟区域边界属于下一次**只用第一条开发序列**冻结的计算细节；不能看第二条人工真值／检测表现后再选阈值。第二条与第一条场景高度相似，即使得到正面结果，也只能说明相似实验条件下的重复性，不能宣称真实园区误入检测。模型的二维人员框成绩与深度区域规则输出分别列示；没有独立三维人员真值，也没有真实危险区标签。

## 下一步需要的真实输入

项目团队两名成员各自完成 115 帧初标／24 帧独立复标，提交两份**未参考检测器输出**的 JSON。收到以后审查分歧并冻结处理规则，然后才运行 A2 模型和视觉接口实验；在此之前，论文不能出现“人员检测精度”或“真实越界准确率”的具体数字。
