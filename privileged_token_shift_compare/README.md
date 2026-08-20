# Privileged Token Shift Compare

独立的离线分析工具，用来回答一个明确的问题：学生模型已经生成固定的
token ID 序列后，同一串 token 在原始多模态条件和文本特权条件下的概率
如何变化。

工具不导入 `verl`，但复现了当前训练链路里相关的数据处理约定：

- 输入是 VERL 风格 JSONL，默认读取 `prompt`、`images` 和
  `reward_model.ground_truth`。
- student 使用原始 prompt 和原图，只生成一次。
- teacher 使用 standalone 文本 prompt，默认模板为
  `{privileged_text} 请复写上面的内容。`，不读取图片。
- student 和 teacher 随后都以 `T=1` teacher forcing 计算这组**完全相同的
  response token IDs**，不会让 teacher 重新生成另一份答案。
- 生成时默认 `temperature=0.6`、`top_p=1.0`，与当前实验设置对齐。
- EOS 保留在响应 token 中，便于单独检查结束概率偏移。

## 输出结构

```text
report/
├── index.html                  # 双击即可打开的批次摘要
├── manifest.js                 # file:// 可直接加载的批次数据
├── manifest.json               # 同一份数据的原始 JSON
├── config.json
├── assets/
└── samples/
    └── <sample>/
        ├── index.html          # 一个样本一个页面，双击可打开
        ├── data.js             # file:// 可直接加载的样本数据
        ├── data.json           # 同一份数据的原始 JSON
        └── input_00.webp
```

单样本页面包含完整生成序列、逐 token 热力图、student/teacher 的概率、
logprob、rank、entropy、top-k 候选、上下文窗口、EOS 标记，以及“相同文本、
不同 token 切分”的候选提示。长序列按 320 token 分批渲染，不会一次创建
全部 DOM 节点。HTML 使用相对路径读取同目录 JavaScript 数据文件，因此
`file://` 模式可以正常工作，不需要部署 HTTP 服务。

## 运行

当前实验配置可直接参考：

```bash
bash run_example.sh
```

也可以显式指定参数：

```bash
python3 compare.py run \
  --dataset /path/to/train.jsonl \
  --model /path/to/Qwen3.5-2B \
  --output /path/to/report \
  --sample-count 8 \
  --selection random \
  --privileged-key reward_model.ground_truth \
  --privileged-template '{privileged_text} 请复写上面的内容。' \
  --temperature 0.6 \
  --max-new-tokens 1024
```

按样本 ID 精确选择时可重复传入：

```bash
python3 compare.py run ... --sample-id sample_a --sample-id sample_b
```

任务支持断点恢复。相同样本、模型和分析参数对应的 `data.json` 已存在时会
直接复用；增加 `--no-resume` 可强制重算。

## 查看页面

运行结束后，直接双击输出目录中的 `index.html`，或者把它拖入浏览器。
不需要运行 Python 服务。请保留整个输出目录，不要只复制单独的 HTML，
因为页面会读取相邻的 `manifest.js`、`data.js`、样式和图片文件。

也可以直接双击任意 `samples/<sample>/index.html` 查看单条样本。

## 关键阈值

- `--student-high 0.40`：student 对目标 token 的高置信阈值。
- `--teacher-low 0.01`：teacher 对目标 token 的低置信阈值。
- `--probability-ratio 10`：标注显著偏好的概率倍率。
- `--teacher-uncertain-entropy 3.0`：区分 teacher 不确定与明确反对。
- `--top-k 10`：每个位置保存的候选数量。

这些阈值只影响报告分类和标记，不改变模型输出或概率计算。

## 依赖

复用 `ICLR_val` 根目录环境即可，核心依赖为 Python 3.10+、PyTorch、
Transformers 5.3+、Pillow 和 `qwen-vl-utils`。工具本身不依赖 `verl`。
