# qwen-mm-privileged-probe 服务器复现指南

本文档用于复现 `qwen-mm-privileged-probe` 的离线 Hugging Face
Transformers 实验。对应双模型教师接口提交：

```text
d4ef1dd Add optional teacher model to privileged probe
```

实验不调用 OpenAI API、vLLM API，也不训练模型。学生模型只生成一次，随后使用
固定的学生 response token ID 序列，在原始条件和特权条件下分别做 teacher
forcing forward。

## 1. 实验口径

对每个样本严格执行以下步骤：

1. 学生模型输入 `图片 + OCR prompt`，贪心生成一次，得到固定的
   `response_ids = [r_1, ..., r_T]`。
2. 将同一组 `response_ids` 直接拼接到学生的 `图片 + OCR prompt` token 后，
   用学生模型 forward，得到 `p_original(r_t)`。
3. 构造包含完整 Markdown GT 的文本特权 prompt。
4. 将原始学生 `response_ids` 直接拼接到特权 prompt token 后，用教师模型
   forward，得到 `p_teacher(r_t)`。
5. 逐 token 保存概率、log 概率、rank、熵、Top-1/Top-2 以及
   `teacher - original` 的变化。

只有第 1 步调用 `generate`。教师模型不重新生成、不采样，也不会先把学生
response 解码后再 tokenizer。response 文本只用于展示和 tokenizer 兼容性校验，
实际 forward 使用的是原始学生 token ID。

核心量定义为：

```text
delta_p    = p_teacher(response_token) - p_original(response_token)
delta_logp = log p_teacher(response_token) - log p_original(response_token)
```

`delta_logp > 0` 表示教师在特权条件下更认可该学生 token，`delta_logp < 0`
表示教师在特权条件下更不认可该学生 token。

## 2. 单模型与双模型

默认不传 `--teacher-model-id`，学生模型同时作为教师模型，只加载一个模型实例。
此时模型参数相同，概率变化主要反映输入条件从图片 OCR 上下文变为 GT 特权上下文。

传入 `--teacher-model-id` 后：

- `--model-id` 或 `--student-model-id` 指定学生模型；
- 学生负责唯一一次生成和原图条件概率；
- 教师只负责特权条件概率；
- 两个模型同时驻留；
- 两者共同继承 `--dtype`、`--device-map` 和 `--trust-remote-code`。

双模型模式下，`delta_logp` 同时包含模型参数差异和上下文差异，不能将其解释为
纯粹的 GT 特权信息因果效应。只有单模型模式固定了模型参数。

程序会在教师 forward 前校验学生和教师 tokenizer 对每个 response token ID
以及完整 ID 序列的解码是否一致。不一致时直接报错，不会自动重分词，因为自动
重分词会破坏“比较同一 token ID 序列”的实验定义。

## 3. 服务器目录变量

以下路径对应当前服务器约定。若实际目录不同，只修改本节变量。

```bash
export PROJECT_ROOT=/home/ma-user/work/wangbaode/03_innovate/ICRL_verify
export DATA_BASE="$PROJECT_ROOT/exp_v2/data"
export DATASET_ARCHIVE="$DATA_BASE/arxiv_confusable_v10_36_server.tar.gz"
export DATASET_ROOT="$DATA_BASE/arxiv_confusable_v10_36_server"

# 可以是 Hugging Face model ID，也可以是服务器本地模型目录。
export STUDENT_MODEL=Qwen/Qwen3.5-4B

# 仅双模型实验需要修改。
export TEACHER_MODEL=/path/to/teacher-model
```

建议所有输出保存在代码目录外可长期保留的位置，或者至少使用独立实验名：

```bash
export OUTPUT_BASE="$PROJECT_ROOT/outputs"
```

## 4. 环境安装

进入上传后的代码目录：

```bash
cd "$PROJECT_ROOT"
```

推荐使用独立 Conda 环境：

```bash
conda create -n qwen-mm-probe python=3.11 -y
conda activate qwen-mm-probe
python -m pip install --upgrade pip setuptools wheel
```

如果服务器需要与 CUDA 驱动匹配的特定 PyTorch，请先按照服务器环境安装 PyTorch，
然后安装本项目：

```bash
python -m pip install -e .
```

确认命令入口和关键依赖：

```bash
qwen-mm-privileged-probe --help

python -c "import torch, transformers, qwen_vl_utils; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('cuda=', torch.cuda.is_available())"
```

代码至少应包含提交 `d4ef1dd`：

```bash
git rev-parse --short HEAD
git log --oneline -5
```

## 5. 数据准备与校验

若数据仍是压缩包：

```bash
mkdir -p "$DATA_BASE"
tar -xzf "$DATASET_ARCHIVE" -C "$DATA_BASE"
```

数据根目录必须直接包含 `pairs.jsonl`：

```text
arxiv_confusable_v10_36_server/
├── pairs.jsonl
└── data/
    ├── example.png
    ├── example.md
    └── ...
```

先进行只读检查：

```bash
test -f "$DATASET_ROOT/pairs.jsonl"
wc -l "$DATASET_ROOT/pairs.jsonl"
head -n 1 "$DATASET_ROOT/pairs.jsonl"
```

`pairs.jsonl` 每行至少需要以下字段：

```json
{
  "pair_id": "sample_0001",
  "edited_image": "data/sample_0001.png",
  "edited_markdown": "data/sample_0001.md",
  "changes": [
    {
      "origin_ans": "原始词",
      "ocr_ans": "图片和编辑后GT中的变异词",
      "bbox": [100, 200, 180, 230],
      "markdown_span": [42, 45]
    }
  ]
}
```

路径可以是相对于 `pairs.jsonl` 所在目录的相对路径。程序会验证图片和 Markdown
GT 是否存在。`changes` 用于变异词对齐和教师信号审计；没有变异标注时仍可生成
逐 token 概率报告，但不会产生有效的变异词统计。

## 6. 固定 Prompt

### 6.1 学生 OCR prompt

若不传 `--prompt` 或 `--prompt-file`，程序使用代码中的
`DEFAULT_PDF_OCR_PROMPT`。为避免不同代码版本的默认值变化，正式实验建议把 prompt
保存为固定文本文件，例如：

```bash
export OCR_PROMPT_FILE="$PROJECT_ROOT/configs/pdf_ocr_prompt.txt"
```

当前默认内容为：

```text
You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with $ $. For example: This is an inline formula $E = mc^2$
- Enclose block formulas with $$ $$. For example: $$\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

3. Table Processing:
- Convert tables to HTML format.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
```

使用固定文件时，在运行命令中增加：

```bash
--prompt-file "$OCR_PROMPT_FILE"
```

### 6.2 教师特权 prompt

每个样本会将完整 Markdown GT 原样放入以下模板：

```text
Please transcribe the document enclosed by the boundary markers verbatim, character by character and symbol by symbol. This is a transcription task, not a translation task. Do not change, correct, add, or omit any character. Output only the document content; do not include the boundary markers.

<<<DOCUMENT_START>>>
{privileged_text}
<<<DOCUMENT_END>>>
```

如果 GT 末尾没有换行，程序只会在结束边界前补一个换行，保证结束边界独占一行。
教师当前接收的是纯文本特权 prompt，不接收原始图片。

## 7. Smoke Test

先用 2 个样本验证环境、模型和 tokenizer。smoke test 使用独立输出目录，避免与
正式实验混用：

```bash
export SMOKE_OUTPUT="$OUTPUT_BASE/qwen35_privileged_probe_smoke"

qwen-mm-privileged-probe \
  --student-model-id "$STUDENT_MODEL" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$SMOKE_OUTPUT" \
  --max-new-tokens 1024 \
  --top-k 5 \
  --forward-chunk-size 16 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --seed 7 \
  --limit 2 \
  --heartbeat-seconds 30
```

若使用固定 OCR prompt 文件，在命令中加入 `--prompt-file "$OCR_PROMPT_FILE"`。

检查 smoke test：

```bash
cat "$SMOKE_OUTPUT/run_summary.json"
cat "$SMOKE_OUTPUT/config.json"
find "$SMOKE_OUTPUT/samples" -name result.json -print
```

应满足：

```text
generation_count_per_sample = 1
response_ids_directly_concatenated = true
response_text_retokenized = false
student_model_id = 指定的学生模型
teacher_model_id = 指定的教师模型，或与学生模型相同
```

## 8. 正式单模型实验

单模型模式最适合分析“同一个模型加入 GT 特权信息后，对原 response token 的认可度
如何变化”。不传 `--teacher-model-id`：

```bash
export SINGLE_OUTPUT="$OUTPUT_BASE/qwen35_privileged_probe_single_model"

qwen-mm-privileged-probe \
  --student-model-id "$STUDENT_MODEL" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$SINGLE_OUTPUT" \
  --max-new-tokens 4096 \
  --top-k 5 \
  --forward-chunk-size 16 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --seed 7 \
  --teacher-signal-threshold 0.05 \
  --heartbeat-seconds 30
```

此时 `config.json` 中应有：

```json
{
  "teacher_model_is_student": true
}
```

## 9. 正式双模型实验

确认两个模型 tokenizer 对学生 response ID 兼容，并确保有足够显存或 CPU
offload 空间：

```bash
export DUAL_OUTPUT="$OUTPUT_BASE/qwen35_privileged_probe_dual_model"

qwen-mm-privileged-probe \
  --student-model-id "$STUDENT_MODEL" \
  --teacher-model-id "$TEACHER_MODEL" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$DUAL_OUTPUT" \
  --max-new-tokens 4096 \
  --top-k 5 \
  --forward-chunk-size 16 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --min-pixels 2048 \
  --max-pixels 16777216 \
  --image-patch-size 16 \
  --seed 7 \
  --teacher-signal-threshold 0.05 \
  --heartbeat-seconds 30
```

此时 `config.json` 中应有：

```json
{
  "teacher_model_is_student": false
}
```

注意：两个模型当前使用同一个 `--device-map` 参数，并同时驻留。如果
`--device-map auto` 仍然 OOM，需要增加可用 GPU/CPU 内存、使用更小模型，或先运行
单模型实验。当前程序不会在每个样本之间反复卸载和加载教师模型。

## 10. 断点续跑

默认启用 resume。若程序被中断，使用完全相同的输出目录和命令重新运行即可：

```bash
qwen-mm-privileged-probe ... --output-dir "$SINGLE_OUTPUT"
```

程序按样本保存 `partial.json` 和最终 `result.json`。已经完成且 fingerprint 一致的
样本会跳过；更换学生模型、教师模型、prompt、GT、生成长度或关键推理参数会改变
fingerprint，从而重新计算相应样本。

`--no-resume` 会忽略已完成结果并重新计算。正式复现实验更推荐使用新的输出目录，
避免不同配置混在一起。

## 11. 只重建报告

若 `result.json` 已经存在，只修改审计阈值或重新生成 HTML/CSV 时，不需要加载模型：

```bash
qwen-mm-privileged-probe \
  --output-dir "$SINGLE_OUTPUT" \
  --teacher-signal-threshold 0.05 \
  --rebuild-report-only
```

`--teacher-signal-threshold` 影响变异词教师信号分类，以及正确 token 页面中的明显
压低判定；它不改变任何模型概率。
当前有效信号定义为：

```text
abs(decision_token_delta_logp) > teacher_signal_threshold
```

一个变异词即使对应多个 tokenizer token，也只统计一次；审计使用该变异词第一个
关联 response token 作为 decision token，完整 subtoken 仍保留在结果中。

## 12. 输出结构

根输出目录的重要文件：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整运行配置、student/teacher 模型 ID、prompt 和推理参数 |
| `run_summary.json` | 完成、跳过、失败和中断状态 |
| `summary.json` | 全局 token 与变异词摘要 |
| `report.html` | 全部样本入口；逐样本查看 GT、response 和所有 token |
| `token_probabilities.csv` | 全部样本的完整 response token 概率 |
| `mutation_probabilities.csv` | 变异词及其关联 response token |
| `sample_summary.csv` | 每个样本的摘要 |
| `teacher_signal_audit.html` | 只分析人工变异词的教师信号质量页面 |
| `teacher_signal_audit.json` | 教师信号质量统计和阈值扫描 |
| `teacher_signal_mutations.csv` | 每个变异词一行的主要审计表 |
| `teacher_signal_tokens.csv` | 仅包含变异词关联 subtokens 的辅助表 |
| `teacher_signal_sample_summary.csv` | 每个样本的变异词教师信号摘要 |
| `correct_token_teacher_rejection.html` | 正确内容 token 与格式 token 的教师不认可审计 |
| `correct_token_teacher_rejection.json` | 概率压低、Top-1 拒绝、分组与阈值扫描统计 |
| `correct_token_teacher_rejection.csv` | 全部纳入 token 的逐行明细 |
| `correct_token_teacher_rejection_sample_summary.csv` | 每个样本的正确 token 教师不认可摘要 |
| `failures.jsonl` | 失败样本及异常信息，仅发生失败时出现 |

每个 `samples/<ordinal>_<pair_id>/` 目录包含：

| 文件 | 内容 |
|---|---|
| `input.<ext>` | 本次实际读取的图片副本 |
| `ground_truth.md` | 完整 Markdown GT |
| `privileged_prompt.txt` | 本样本实际使用的教师 prompt |
| `response.md` | 学生唯一一次生成的文本 |
| `response_ids.json` | 学生原始 response token ID 序列 |
| `result.json` | 完整协议、概率、Top-K、对齐和变异词结果 |
| `token_probabilities.csv` | 该样本所有 response token，严格保持生成顺序 |
| `mutation_probabilities.csv` | 该样本变异词结果 |
| `report.html` | 该样本可视化页面 |

## 13. 结果字段解释

逐 token 主要字段：

| 字段 | 含义 |
|---|---|
| `token_id` | 学生生成的原始 token ID |
| `raw_token` | 该 token 的原始解码文本 |
| `p_original` | 学生模型在图片 + OCR prompt 下对该 token 的概率 |
| `rank_original` | 该 token 在学生原始分布中的排名 |
| `top_candidates_original` | 学生原始分布的 Top-K |
| `p_teacher` | 教师模型在 GT 特权 prompt 下对同一 token ID 的概率 |
| `rank_teacher` | 同一 token ID 在教师分布中的排名 |
| `top_candidates_teacher` | 教师分布的 Top-K |
| `delta_p_teacher_minus_original` | `p_teacher - p_original` |
| `delta_logp_teacher_minus_original` | `logp_teacher - logp_original` |
| `top1_changed` | 原始条件与教师条件的 Top-1 token ID 是否变化 |

变异词主要字段：

| 字段 | 含义 |
|---|---|
| `origin_ans` | 变异前字符或词 |
| `ocr_ans` | 编辑后图片和 GT 中实际出现的变异词 |
| `predicted` | 学生模型对该位置的读回结果 |
| `relation=expected` | 学生读回与编辑后 GT 一致 |
| `relation=opposite_variant` | 学生读成了变异前内容 |
| `relation=other` | 学生读成第三种内容 |
| `relation=deleted` | 学生 response 中没有对应内容，无法对固定 response token 打分 |
| `decision_delta_logp` | 该变异词第一个关联 response token 的 `delta_logp` |

正确 token 教师不认可页面只纳入 `token_label=correct` 和
`token_label=formatting`，并分别报告：任意概率下降、超过阈值的概率压低、Teacher
Top-1 token ID 不同、Teacher Top-1 解码文本不同。格式 token 按要求纳入，但空格、
换行和 Markdown 格式在 OCR 对齐标准化时会被移除，因此其“正确”状态没有经过与
普通内容 token 相同的字符级 GT 验证。

## 14. 在本地查看服务器结果

服务器没有浏览器时，可以将整个输出目录下载到本地后直接打开 `report.html` 和
`teacher_signal_audit.html`。

也可以在服务器启动只读 HTTP 服务：

```bash
cd "$SINGLE_OUTPUT"
python -m http.server 8000 --bind 127.0.0.1
```

在本地建立 SSH 转发：

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```

然后本地浏览器访问：

```text
http://127.0.0.1:8000/report.html
http://127.0.0.1:8000/teacher_signal_audit.html
http://127.0.0.1:8000/correct_token_teacher_rejection.html
```

## 15. 建议保存的复现信息

正式运行前创建输出目录并记录环境：

```bash
mkdir -p "$SINGLE_OUTPUT/repro"
git rev-parse HEAD > "$SINGLE_OUTPUT/repro/git_commit.txt"
python --version > "$SINGLE_OUTPUT/repro/python_version.txt" 2>&1
python -m pip freeze > "$SINGLE_OUTPUT/repro/pip_freeze.txt"
nvidia-smi > "$SINGLE_OUTPUT/repro/nvidia_smi.txt"
```

同时保留以下内容：

- 完整启动命令和 shell 日志；
- `config.json`；
- 模型目录或 Hugging Face revision；
- 数据压缩包 SHA256；
- 代码 commit；
- PyTorch、Transformers、CUDA 和驱动版本。

可记录数据校验值：

```bash
sha256sum "$DATASET_ARCHIVE" > "$SINGLE_OUTPUT/repro/dataset_sha256.txt"
```

## 16. 常见问题

### 命令不存在

确认已经在当前环境执行：

```bash
cd "$PROJECT_ROOT"
python -m pip install -e .
```

### qwen-vl-utils 缺失

Infinity-Parser/Qwen-VL 风格图片预处理需要 `qwen-vl-utils`。执行：

```bash
python -m pip install "qwen-vl-utils>=0.0.14"
```

### 双模型 tokenizer 不兼容

若提示同一个 response token ID 在 student/teacher 下解码不同，说明不能直接比较
这两个模型的原始 token ID。应换成共享 tokenizer 和 ID 映射的教师模型。不要通过
解码后重新 tokenizer 绕过校验，否则比较对象已经改变。

### 双模型 CUDA OOM

双模型会同时驻留。先确认 `--device-map auto`，检查 GPU/CPU 可用内存；必要时改用
单模型、较小教师模型或增加设备资源。

### 没有生成报告

检查：

```bash
cat "$SINGLE_OUTPUT/run_summary.json"
test -f "$SINGLE_OUTPUT/failures.jsonl" && tail -n 20 "$SINGLE_OUTPUT/failures.jsonl"
find "$SINGLE_OUTPUT/samples" -name result.json | wc -l
```

至少需要一个完整 `result.json` 才能构建根报告。

### 修改审计阈值后是否需要重新推理

不需要。使用 `--rebuild-report-only` 即可。阈值只改变审计分类，不改变已保存概率。

### `--forward-chunk-size` 是否把模型 forward 分块

不是。固定 response 的模型 forward 仍按完整上下文执行；该参数主要控制后续逐
token 概率和 Top-K 计算的处理块大小。

## 17. 最终复现检查清单

- 代码包含提交 `d4ef1dd`。
- `pairs.jsonl`、图片和 Markdown GT 路径均有效。
- 正式命令保存了 student/teacher 模型 ID 和 revision。
- 学生 `generation_count` 为 1。
- `response_ids_directly_concatenated` 为 `true`。
- `response_text_retokenized` 为 `false`。
- 单模型实验 `teacher_model_is_student` 为 `true`。
- 双模型实验 `teacher_model_is_student` 为 `false`。
- 每个 token 的原始顺序与 `response_ids.json` 一致。
- 变异词审计只统计 `changes` 中标注的人工变异词。
- 报告、配置、环境版本、日志和数据 SHA256 均已归档。
