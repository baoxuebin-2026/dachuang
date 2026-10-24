# 论文数据下载与实验结果索引

论文使用 UCI 361、UCI 270 和 UCI 487。原始压缩包合计约 380 MB，不纳入 Git 历史，避免普通克隆自动下载大文件。文件名、官方下载地址、字节数、DOI 和 SHA-256 记录在 `public_datasets_manifest.json`。仓库保留下载脚本、冻结协议、运行代码、派生特征、逐条预测和汇总结果，换电脑后可以重新下载并校验原始数据。

## 1 下载并校验原始数据

在仓库根目录运行：

```bash
python scripts/download_paper_datasets.py
```

文件默认保存到 `data/raw/`，该目录已被 `.gitignore` 排除。如果下载中断，删除相应的 `.part` 文件后重新运行。无论从 UCI 还是其他镜像下载，都应以清单中的字节数和 SHA-256 为准。

安装论文实验所需依赖：

```bash
python -m pip install -r requirements-eda.txt
```

## 2 三套数据分别检验什么

| 数据集 | 原始内容 | 本文任务 | 作用 |
|---|---|---|---|
| UCI 361 | 5 台完整设备、每台 8 通道、4 类气体、10 个浓度档 | 其他设备第 4—10 档训练，留出设备第 1—3 档测试 | 主实验：同时检验换设备、浓度下降和四气体识别 |
| UCI 270 | 1 套 16 通道阵列、6 类气体、10 个时间批次、13 910 次测量 | 批次 1—5 高浓度训练，批次 6—10 低浓度测试 | 压力测试：检验长期漂移，不含 CO，也不是多设备复现 |
| UCI 487 | 14 只温度调制传感器、13 日、1 300 个配气段 | 7 只同型号传感器轮流留出，前 10 日高浓度训练、后 3 日低浓度测试 | 压力测试：检验跨传感器低浓度 CO，但湿度、日期和加热相位同时变化 |

## 3 运行方式

### UCI 361 主实验审计

主实验的逐文件预测已经保存在仓库，可先直接复核论文数字，不需要重新拟合模型：

```bash
python experiments/uci361_low_concentration/paper_preflight_2026_10_22.py
```

如需从官方压缩包完整恢复主实验，先复制为脚本约定的文件名，再运行恢复和一致性检查：

```bash
cp data/raw/uci_361_twin_gas_sensor_arrays.zip experiments/uci361_low_concentration/source.zip
OPENBLAS_NUM_THREADS=1 python experiments/uci361_low_concentration/uci361_recovered_audit.py
python experiments/uci361_low_concentration/uci361_verify_recovery.py
```

Windows PowerShell 可将 `cp` 改为 `Copy-Item`，并把第一条运行命令写成：

```powershell
$env:OPENBLAS_NUM_THREADS="1"
python experiments/uci361_low_concentration/uci361_recovered_audit.py
```

### UCI 270 长期漂移压力测试

Linux 或 macOS：

```bash
UCI270_ARCHIVE=data/raw/uci_270_gas_sensor_array_drift.zip \
python experiments/uci270_high_to_low_stress/uci270_amplitude_stress.py
```

Windows PowerShell：

```powershell
$env:UCI270_ARCHIVE="data/raw/uci_270_gas_sensor_array_drift.zip"
python experiments/uci270_high_to_low_stress/uci270_amplitude_stress.py
```

结果写入 `experiments/uci270_high_to_low_stress/uci270_amplitude_stress.json`。

### UCI 487 跨传感器低浓度 CO 压力测试

```bash
python experiments/uci361_low_concentration/uci487_cross_device_2026_10_23.py \
  --archive data/raw/uci_487_gas_sensor_array_temperature_modulation.zip \
  --protocol experiments/uci361_low_concentration/uci487_cross_device_protocol_2026_10_23.json \
  --out results/uci487_cross_device_2026_10_23
```

## 4 已保存结果

| 数据与方法 | 原始 SVM 宏平均 F1 | 幅值增强 SVM 宏平均 F1 | 其他结果 |
|---|---:|---:|---|
| UCI 361，5 台设备、4 类气体 | 0.736 | **0.963** | 幅值增强 CNN 为 0.974，187/192 次正确，CO 48/48 |
| UCI 270，跨批次 6 类气体 | 0.225 | **0.452** | 5 个测试批次均提高，但绝对结果偏低 |
| UCI 487，跨器件 CO/空气 | 0.529 | **0.593** | 15/21 个评价块提高，CO 平均召回率 0.512 |

关键结果文件：

- UCI 361：`experiments/uci361_low_concentration/paper_preflight_2026_10_22.json`、`paper_normalization_control_2026_10_22.json`、`uci361_recovered_audit_2026-10-20.json` 及相应预测 CSV。
- UCI 270：`experiments/uci270_high_to_low_stress/uci270_amplitude_stress.json`。
- UCI 487：`results/uci487_cross_device_2026_10_23/`，其中包含派生特征、逐条预测、21 个评价块结果、汇总 CSV 和运行记录 JSON。
- 论文图表：`论文结果图_期刊重构版_2026-10-24/`，每张图同时提供 PNG、SVG 和 PDF，绘图代码为 `experiments/uci361_low_concentration/plot_manuscript_journal_2026_10_24.py`。

三套数据没有拼接训练，也没有计算跨数据集总准确率。UCI 270 和 UCI 487 用于说明方法适用边界，不能当作与 UCI 361 完全相同任务的独立复现。

## 5 GitHub Release 可选方案

仓库包含 `.github/workflows/publish-paper-datasets.yml`。推送 `paper-datasets-v*` 标签时，GitHub Actions 会从 UCI 官方下载、校验，并把三套原始压缩包发布为 Release 附件。普通 `git clone` 不会下载这些原始压缩包。
