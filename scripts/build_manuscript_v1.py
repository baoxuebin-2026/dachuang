#!/usr/bin/env python3
"""Build a standalone manuscript v1 from the reviewed section drafts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def from_heading(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    pos = text.index(heading)
    return text[pos:].strip()


def main() -> None:
    front = """# 异源感知条件下面向焦化园区的分级预警方法研究——证据质量门控与模块化验证

> 投稿工作稿 v1，2026-09-27。作者、单位、基金编号和通信信息待项目组与指导教师确认。本稿未投稿。

## 摘要

针对焦化园区拟部署预警中气体与视觉证据来源不同、时间和区域不可直接对齐以及缺测证据易被误用的问题，提出一种带来源、时钟、区域、质量、时效和缺测原因的异源证据接口及分级处置方法。方法分别在 UCI 309 风洞气体数据、Bonn 室内 RGB-D 数据和 UCI 309 气体证据加仿真人员轨迹的 S1 场景中验证，并以 METEC 受控释放数据检查固定阈值的跨日期稳定性。UCI 309 的八通道中位规则在 60 个测试文件中有 57 个于供气后 60 s 内形成持续响应，释放前触发 1 个文件；Bonn 115 帧中匹配 62/66 个人员框，第二段 34 个有人帧中 32 个得到有效检测框深度；四类各 600 s 的故障重放中，完整门控在不满足配对条件的窗口输出联合未知且无 L3，移除相应门控后产生 550 或 553 s 的无效配对 L3；METEC 按日期留一时，训练背景 P95 固定阈值对释放活动观测秒的覆盖中位数为 18.0%，显示明显日期差异。结果表明，显式质量和时空门控可使异源证据的联合条件可审计，单点固定阈值不宜脱离日期和环境上下文直接迁移。本文验证的是公开数据局部功能和模拟规则行为，不代表焦化园区现场事故预警准确率或硬件性能。

**关键词：**焦化园区；气体传感器；RGB-D；异源感知；质量门控；分级预警；模块化验证
"""
    parts = [
        front.strip(),
        from_heading(DOCS / "manuscript-section-1-v1.md", "## 1 引言"),
        from_heading(DOCS / "manuscript-sections-2-4-v1.md", "## 2 研究问题与证据边界"),
        from_heading(DOCS / "manuscript-sections-5-7-v1.md", "## 5 实验结果"),
        """## 数据与代码可得性

分析代码、冻结配置、汇总结果、证据账本和绘图脚本见项目 GitHub 仓库。UCI、Bonn 和 METEC 原始数据按各发布方许可从原站获取，原始大文件未重复分发。正式投稿时补充匿名或公开仓库链接及版本提交号。

## 致谢与基金项目

待确认山西省高等学校大学生创新训练计划项目编号、项目名称、作者贡献和指导教师信息后填写。""",
    ]
    out = DOCS / "manuscript-full-v1.md"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
