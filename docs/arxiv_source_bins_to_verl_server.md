# 从 arXiv `source_archive.bin` 批量生成 VERL 数据

入口程序：`scripts/run_arxiv_source_bins_to_verl.py`

## 输入目录

输入必须是 `scripts/crawl_arxiv_sources.py` 的完整输出，而不是一批脱离元数据的 `.bin`：

```text
download_root/
├── results.jsonl
└── papers/
    └── <arxiv-id><version>/
        └── source_archive.bin
```

程序会重新核对论文 ID、版本、许可白名单、归档存在性、非空状态和下载阶段记录的 SHA-256。只接受 `CC-BY-4.0`、`CC-BY-SA-4.0` 和 `CC0-1.0`。

## Linux 环境

需要完整 TeX 环境、Poppler 和 Python 包：

```bash
sudo apt-get update
sudo apt-get install -y latexmk poppler-utils ghostscript texlive-full

python -m pip install -r requirements-arxiv-source-bins-to-verl.txt
```

`texlive-full` 占用空间较大，但面对来源多样的 arXiv 宏包时成功率最高。服务器已有完整 TeX Live 时不需要重复安装。

实际检查命令：

```bash
python scripts/check_arxiv_source_bins_to_verl_environment.py \
  --require-all-engines \
  --json-report arxiv_verl_environment_report.json
```

该命令不仅查找可执行文件，还会实际完成三条最小编译链并调用 `pdftoppm` 生成 PNG。最终出现下面一行才表示完整环境通过：

```text
[finish] status=passed passed_engines=pdflatex,xelatex,latex_dvips_ps2pdf ...
```

## 批量命令

```bash
PYTHONUNBUFFERED=1 python scripts/run_arxiv_source_bins_to_verl.py \
  --input-root /data/arxiv_sources_2000 \
  --work-root /data/arxiv_source_first_work \
  --output-dir /data/arxiv_confusable_verl_s83 \
  --workers 8 \
  --seed 83 \
  --split-seed 42 \
  --val-fraction 0.05 \
  --dpi 144 \
  --compile-timeout 600 \
  --paper-timeout 2400 \
  --latexmk "$(command -v latexmk)" \
  --pdftoppm "$(command -v pdftoppm)"
```

默认处理全部合格归档。试跑前 20 篇：

```bash
python scripts/run_arxiv_source_bins_to_verl.py \
  --input-root /data/arxiv_sources_2000 \
  --work-root /data/arxiv_source_first_smoke_work \
  --output-dir /data/arxiv_confusable_smoke \
  --max-papers 20 \
  --workers 4
```

指定论文：

```bash
... --paper-ids 2606.00044v1 2606.00856v1
```

## 并行和资源

- 源码安全解压与 source-first GT：按论文使用 `ProcessPoolExecutor`。
- 变异源码重编译：按论文使用独立进程。
- 主进程至少每 30 秒输出全局完成数、页面数、字节数、百分比、吞吐率、耗时、ETA、接受/拒绝/错误数。
- 每个 TeX 编译本身可能继续派生 TeX 子进程，因此不要直接把 `--workers` 设为全部逻辑 CPU。建议从 `min(8, 物理核数/2)` 开始。
- 受限平台若禁止进程信号量，程序会明确打印警告并回退到线程；普通 Linux 服务器使用真正的多进程。

## 编译回退

每篇论文默认依次尝试：

```text
pdflatex,xelatex,latex_dvips_ps2pdf
```

可通过下面参数改变顺序或只保留一种：

```bash
--latex-engines pdflatex,xelatex
```

源码提供的脚本、Makefile 和 `latexmkrc` 不会执行，shell escape 被关闭。检测到危险构造的论文会被拒绝并保留审计记录。

## 断点恢复

默认开启恢复：

- 每篇论文的解压、安全扫描、source-first GT 和变异结果均独立保存。
- 已通过论文直接复用。
- 失败论文默认不反复重试；修复环境后使用 `--retry-failed`。
- 使用 `--no-resume` 禁用恢复。
- 不完整的源码或 source-first 输出会移动到该论文的 `diagnostics/`，不会直接删除。

重复运行同一命令即可继续。

## 最终输出

```text
output_dir/
├── data/                         # 变异后的整页 PNG
├── ground_truths/                # 与页面严格对应的变异 Markdown GT
├── metadata/                     # 变异词和 bbox
├── SFT_edited_<N>.jsonl
├── verl_grpo/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── train.parquet
│   └── val.parquet
├── validation_report.json
├── independent_verifier_report.json
└── pipeline_report.json
```

只有 `validation_report.json` 和 `independent_verifier_report.json` 均为 `passed` 时，`pipeline_report.json` 才会写入 `status: passed`。

SFT 和 VERL 中的图片路径就是实际生成文件的绝对路径：

```text
<output-dir>/data/<pair-id>_edited.png
```

`--output-dir` 是唯一的数据集根目录，不再存在独立的 `--server-root` 路径重写。
默认按服务器直接生成和训练处理；如果需要在本地生成后上传，应在上传完成后显式重写并重新校验训练清单。
