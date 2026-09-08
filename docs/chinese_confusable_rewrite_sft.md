# 中文形近字 rewrite SFT：直接输出最终训练数据

入口：`scripts/build_chinese_confusable_rewrite_sft.py`。
独立脚本，不导入旧 arXiv 脚本、不需要 LaTeX、PDF、图像或模型推理。
Python 3.10+；第三方依赖只有 `tokenizers`。只加载模型目录中的
`tokenizer.json`，不加载模型权重。

## 一条命令生成

在服务器的代码仓库根目录执行：

```bash
python -u scripts/build_chinese_confusable_rewrite_sft.py \
  --output-dir /inspire/sfs/project/inf-multimodal/public/wangbaode/06_datasets/04_teacher_force_data/chinese_rewrite_v1 \
  --tokenizer /home/ma-user/work/share_base_models/Qwen3.5/Qwen3.5-2B/ \
  --workers 128
```

默认下载 D2L 中文和 OI Wiki 的 Markdown 正文文件，边下载、处理边输出；
不会下载仓库中的 PDF、图片、模型、完整 Git 历史，也不保存下载正文副本。
GitHub 文件清单及 commit 固定在 `.state/public_sources.json`，重启使用同一版本。
公共下载同时最多 8 个请求，避免 128 个进程同时请求 GitHub；解析、变异、
token 计数和样本构造使用 `--workers` 个进程。可设置环境变量 `GITHUB_TOKEN`
提高 GitHub API 配额，但不是必需参数。

首次确认可以加 `--max-documents 20`。不设置目标数量时，现有合格语料有多少
就生成多少，不循环复制或补齐数量。这两个教学语料源规模有限，不宣称能产出
百万条；扩量可输入其他有使用权限的中文正文 JSONL。

本地已有中文语料：

```bash
python -u scripts/build_chinese_confusable_rewrite_sft.py \
  --input /path/to/chinese_corpus.jsonl /path/to/markdown_directory \
  --output-dir /path/to/chinese_rewrite_sft \
  --tokenizer /home/ma-user/work/share_base_models/Qwen3.5/Qwen3.5-2B/ \
  --workers 128
```

输入可以是多个 Markdown、TXT、JSONL 文件或目录（递归发现上述扩展名）。
本地 JSONL 按行流式派发任务，不等整个文件扫描或处理完成才生成训练数据；
本地输入按文件名和行序遍历，不额外把整个语料读进内存随机打乱。
JSONL 每行格式：

```json
{"text":"# 标题\n\n中文正文……", "url":"来源地址", "license":"来源许可", "attribution":"原作者"}
```

也接受 `markdown` 字段，存在时优先于 `text`。`url/title/license/license_url/attribution`
会保留到每条样本的来源记录里。路径按启动命令所在目录解释，不擅自改成绝对路径。
输入是文本语料，不是 arXiv `source_archive.bin`、网页抓取 HTML 或 OCR 的结构块 JSON。

## 已确认的 A/B 和 prompt

1. 从同一个原文取连续片段，不拼接不同文档，不重复造多个变体凑数量。
2. 按**未被保护的正文汉字总数**的 2% 四舍五入，做一字换一字，得到 A。
   分母不是“形近字字典中可替换的汉字数”，更不是英文单词数。
3. A 不额外包裹围栏。B 只改变 A 中已有标题的开头 `#` 数量，并添加整篇围栏。
   原等级 1–6；每个已有标题以 28% 概率去掉全部 `#`，剩余 72% 均分给
   1–4 中不同于原等级的候选。原等级为 1–4 时三个候选各占 24%，原等级为
   5–6 时四个候选各占 18%。这是逐标题抽样，不是整篇无标题的样本占比。
   去掉 `#` 时仍保留后面的空格；原本没有标题的文字不添加标题。
4. 空格、换行（包括 CRLF）、正文、标点、数字、变异字及其他格式逐字符保留。
   只有 B 的标题等级和外层围栏不同。不会随机化标题后的空格。

精确 prompt 沿用之前已确认版本（有测试验证两脚本的常量完全相同）：

````text
Please rewrite the document enclosed by the boundary markers using only these two formatting changes. This is not a translation task.
1. For each text block beginning with an existing heading prefix of 1 to 6 # characters, randomly choose a different number of # characters from 0 to 4 (0 means removing all leading # characters). Change only the number of # characters. Preserve the spaces after them exactly. Do not add heading prefixes to non-heading text or change prefixes inside figures, tables, formulas, or code.
2. Enclose the entire result in a Markdown code fence: start with ```markdown followed by a newline, and end with a newline followed by ```.
Preserve every other character exactly, including spelling errors, numbers, whitespace, line breaks, HTML tables, and LaTeX formulas. Do not correct, add, omit, or explain any content. Do not output the boundary markers.

<<<DOCUMENT_START>>>
{A}
<<<DOCUMENT_END>>>
````

答案固定为 `"```markdown\n" + 修改标题等级后的 A + "\n```"`。
不加 system message、不加图片字段，不调用大模型生成答案。
原文已有的内层代码围栏也原样保留；这是一项字面文本重写任务，不额外重排内部围栏。
当前版本为 `chinese_confusable_rewrite_v2_heading_zero`，Prompt 为
`heading_rewrite_boundary_en_v3`。生成新版时使用新输出目录；旧数据不覆盖、
不混用。A 的字符变异种子与旧版一致，标题策略更新只影响 B。

## 形近字与保护范围

初版采用脚本中公开可审计的 40 组双向字形对（80 个方向），例如：
`未↔末、土↔士、己↔已、日↔目、清↔情、待↔侍、微↔徽、辨↔辩`。
不是同音字替换，不替换数字，也不会从所有汉字中随意选一个。
这是初始人工指定候选清单，不声称覆盖所有字形混淆或经过字体相似度验证。

从当前片段可用的变异方向中随机选择，再选该方向的字符位置，同一位置最多变异一次。
可替换位置不足时用实际数量，不为了达到 2% 插入、重复或删除内容；记录实际比例。
完全没有变异的片段不导出到这批变异数据。

保护 HTML 表格/figure/pre/code，Markdown 管道表格，代码围栏及缩进代码，
行内代码，常见 LaTeX 公式定界符和公式环境，HTML 标签/属性、链接目标、URL、
YAML front matter 和引用定义。表格和公式保持原样，不强行增加公式，也不把
Markdown 表格改成 HTML（否则会引入未批准的 A/B 差异）。
现有 Markdown 方言中的标签和指令不做全面标准化。

## 最终输出及持久化

```text
output-dir/
  train.jsonl                 # 实时可训练；不是中间数据
  val.jsonl                   # 实时验证集，同一原文的片段不跨 train/val
  summary.json                # 本次完成/中断后的统计
  .state/
    config.json
    public_sources.json       # 仅公开下载模式
    completed_documents.jsonl
    errors.jsonl
```

每条最终记录只有一个 user 和 assistant，符合 ms-swift 文本 SFT 的 `messages` 格式：

```json
{"messages":[{"role":"user","content":"上述 prompt，其中含变异后的 A"},{"role":"assistant","content":"```markdown\nB 的正文\n```"}],"data_source":"chinese_confusable_text_rewrite","ability":"heading_format_rewrite","extra_info":{"sample_id":"稳定 ID","changes":[],"heading_changes":[],"source":{}}}
```

实际 `changes` 非空，记录 `origin_ans/ocr_ans/input_char_offset/input_char_end`
及在完整 B（包括围栏）里的 `char_offset/char_end`。偏移是 Python Unicode
字符索引，不是 UTF-8 字节或 tokenizer token 索引。另有请求/实际变异数、分母、
实际比例、token 长度、原文地址/commit/许可及片段起止位置。

worker 每完成一个样本立即通知父进程，父进程唯一写入并逐条 flush + fsync。
不等整个文档或整个语料完成，不需要再合并 part 文件。
相同命令重启会复用已完成文档、跳过已有样本 ID；意外中断写了一半的最后一行会移除
后重做，不删除完整记录。并发完成顺序可能不同，但每个样本内容由固定种子决定。
修改变异比例、tokenizer、长度等生成设置时换新的输出目录。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--mutation-ratio` | 0.02 | 正文汉字变异比例 |
| `--min-response-tokens` | 1000 | 完整 B 的最短 token 长度 |
| `--max-response-tokens` | 7800 | 完整 B 的最长 token 长度，包括围栏 |
| `--workers` | CPU 数与 16 的较小值 | 进程数；服务器可设 128 |
| `--max-samples` | 0 | 最终 train+val 样本总数上限，含已保存记录；0 不限 |
| `--max-documents` | 0 | 原文数上限：公开来源固定种子随机抽取，本地取遍历顺序前 N 条；0 用全部 |
| `--seed` | 83 | 抽样、变异、标题选择的种子 |
| `--val-fraction` | 0.02 | 按原始文档哈希分配验证集，小批次可能为 0 条 |

注意：response 约 8k 时，输入也含完整 A，训练的**总上下文长度**需要容纳
prompt + A + B，不能把 ms-swift 的总长度限制误设成只有 8k。
`--tokenizer simple` 只供单元测试，不能用于最终训练数据的 token 长度控制。

## 语料来源和许可

- [D2L 中文](https://github.com/d2l-ai/d2l-zh)，正文版权声明见
  [config.ini](https://github.com/d2l-ai/d2l-zh/blob/master/config.ini)。
- [OI Wiki](https://github.com/OI-wiki/OI-wiki)，非代码部分默认 CC-BY-SA-4.0 + SATA，
  个别内容声明除外；见 [README](https://github.com/OI-wiki/OI-wiki/blob/master/README.md)。

逐条保留来源和署名/许可信息。公开可下载不等于无条件可再分发；发布衍生数据时须遵守
源站适用许可，代码片段及特别声明需按各自许可处理。
没有引入 OmniDocBench 等评测集，没有下载整个 Wikipedia 或 TB 级网络语料。
