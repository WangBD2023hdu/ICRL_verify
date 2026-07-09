# 文档解析方向方法创新型 Paper 选题与写法报告

整理时间：2026-07-07  
目标：围绕 ICLR 主会口味，设计一篇更偏方法创新、而不是单纯 benchmark / dataset 的文档解析论文  
参考基础：前面对 ICLR 2026 文档解析、表格、图表、流程图、scientific diagram、DocQA、retrieval/RAG 相关论文的 OpenReview 复盘

---

## 1. 先给结论

如果目标是做一篇**方法创新型**文档解析 paper，最危险的写法是：

```text
我们把 OCR / parser / retriever / MLLM / agent 组合起来，
在若干 DocQA benchmark 上取得更好结果。
```

这种写法很容易被 reviewer 归为：

- pipeline engineering
- prompt trick
- RAG patchwork
- agent orchestration
- parser wrapper
- benchmark-specific tuning

更像主会的方法创新论文，应该写成：

```text
文档解析暴露出现有 MLLM 的一个基础机制缺陷。
我们提出一个可泛化的结构化机制来修复这个缺陷。
该机制能被形式化、能被消融、能迁移到多个文档任务，
并且相对最直接的强 baseline 有清晰收益。
```

方法创新的核心不是“模块多”，而是**机制清楚**。一个 reviewer 看完后应该能记住一句话：

- DAVE：为 document / web 场景学习专用 vision encoder。
- TableDART：按实例动态路由 text / image / fusion 路径。
- Visual Self-Refine：用 pixel-level visual feedback 修正 chart parsing。
- DaVinci：用 SFT + RL 学习 scientific diagram 的 visual-structural syntax。
- FlowGen：用可控生成系统学习和评估结构复杂度。

这些论文能中，不是因为它们都完美，而是因为它们都有一个可命名、可验证、可消融的核心机制。

---

## 2. 方法创新型论文的主会门槛

### 2.1 必须从应用问题上升到机制问题

不要只说：

```text
文档解析很重要。
现有模型在文档上不好。
我们提出一个框架。
```

要说：

```text
文档解析中的失败来自一个可定位的机制缺陷：
模型缺少结构化证据表示 / 视觉自检能力 / 模态路由能力 / 长文档证据聚合能力。
我们提出一个机制来补这个缺陷。
```

也就是说，论文的中心不能是“document parsing application”，而应该是：

- visual-structural representation
- evidence-grounded reasoning
- adaptive modality routing
- visual self-verification
- structured output decoding
- parser-reasoner co-training
- cost-aware long-document reading

### 2.2 必须能排除 obvious baseline

方法创新型论文最常被问：

- 这不就是 OCR + LLM 吗？
- 这不就是 parser + LLM 吗？
- 这不就是 RAG 吗？
- 这不就是 agent 拼装吗？
- 这不就是换个 prompt format 吗？
- 这不就是 attention pruning 吗？
- 这不就是固定规则 routing 吗？

所以实验一定要直接对比最容易被想到的版本。

例如：

- Adaptive routing 必须比 fixed routing / always fusion / oracle-free heuristic 更好。
- Visual feedback 必须比 text-only self-refine / CoT / direct retry 更好。
- Structured output 必须比 plain Markdown / HTML / OCR text / parser JSON 更好。
- Compression 必须比 fixed-ratio pruning / random pruning / attention top-k 更好。
- Agent system 必须比 strong single-agent RAG 更好。

### 2.3 必须有成本表

文档解析方法常常引入：

- 高分辨率输入
- 多页图像
- OCR
- layout parser
- table/chart parser
- retrieval index
- multi-step reasoning
- self-refine loop
- multi-agent collaboration

所以 reviewer 会自然问：

- latency 增加多少？
- token 增加多少？
- memory 增加多少？
- storage 增加多少？
- 调用次数增加多少？
- 性能提升是否值得？

如果没有成本表，很容易被认为“不实用”。

---

## 3. 从已有论文看“真方法创新”和“伪方法创新”

### 3.1 真方法创新的几个正例

#### DAVE

创新点不是“训练一个 document encoder”这么简单，而是把问题放在 representation learning 上：通用 vision encoder 缺少文档和网页中的结构/空间特征。

可学之处：

- 用 encoder 层创新承接多个任务。
- 不局限在单个 DocQA benchmark。
- 有 document、web、VQA、agent 多类评测。

风险：

- reviewer 仍然质疑它是 MAE、model merging、ensemble training 的组合。
- 所以这类论文必须用非常宽的实验覆盖和 ablation 把贡献撑起来。

#### TableDART

创新点是 instance-level modality routing。它不是把 text/image/fusion 全部混在一起，而是根据 table-query pair 动态选择路径。

可学之处：

- 有清楚机制：routing。
- 有效率收益：避免所有样本都走 fusion。
- 有强 baseline 和 ablation。

风险：

- fusion agent 黑盒。
- 如果没有 routing distribution、failure case、cost analysis，会被认为只是系统拼装。

#### Visual Self-Refine

创新点是把 self-refinement 从文本反思推进到视觉反馈：模型生成 pixel-level localization，再把可视化结果喂回去修正。

可学之处：

- 机制非常好解释。
- 中间状态可视化，可诊断。
- 与 chart parsing 任务高度契合。

风险：

- 方法偏 task-specific。
- 额外 refine calls 带来成本。
- 需要证明不是更强 base model 自然解决的问题。

#### DaVinci

创新点是把 scientific diagram parsing 变成 visual-structural syntax learning，并用 SFT + RL 学习从 raster diagram 到 TikZ code 的结构化映射。

可学之处：

- 输出空间不是普通答案，而是可执行/可渲染代码。
- reward 包含 visual fidelity、structural consistency、code correctness。
- 数据顺序、注释、结构 reward 都可消融。

风险：

- 自动指标和 reward 重叠，可能被质疑 metric hacking。
- 必须做人类评估或外部指标验证。

### 3.2 伪方法创新的几个反例

#### Chain-of-Reading

想法：让模型先定位 evidence，再 OCR，再推理。  
问题：reviewer 认为 locate-then-reason 接近 DocReact / agent workflow / CoT，且缺少强 parser baseline 和成本分析。

教训：

如果你说 end-to-end 优于 pipeline，就必须直接打 MinerU / strong parser + LLM 这类 baseline，并证明 token/latency 成本值得。

#### Structured Attention Matters

想法：用 LaTeX-like structured text 改善文档理解。  
问题：被认为像 formatting trick，缺少 HTML/XML/Markdown baseline，也缺少 causal attention evidence。

教训：

格式输入类方法必须证明机制，不然很容易被归为 prompt engineering。

#### MDocAgent

想法：五个 agent 协作文档 QA。  
问题：被认为是 RAG + prompting + agent 拼装，缺少学习创新、泛化验证、效率分析。

教训：

agent 数量不是贡献。必须证明 agent 分工是必要的，并且比 strong single-agent RAG 更好。

#### DocPruner

想法：adaptive patch-level embedding pruning 降低 VDR 存储。  
问题：没有证明 adaptive threshold 比 fixed-ratio attention pruning 更必要。

教训：

efficiency 方法必须击败最简单强 baseline。否则复杂机制站不住。

---

## 4. 方法创新型文档解析论文的 5 个可行方向

### 4.1 方向 A：Evidence Graph Learning

一句话：

```text
学习一个文档 evidence graph，把 OCR span、layout block、table cell、chart mark、page region、question entity 连接起来，用于可解释文档推理。
```

核心机制：

- 先从文档图像中抽取候选 evidence nodes。
- 节点包括 text span、bbox、table cell、figure region、chart mark、caption、page-level context。
- 预测 relation edges，例如 same-row、same-column、caption-of、refers-to、supports-answer、contradicts、requires-arithmetic。
- 最终用 evidence graph 做 QA、verification、hallucination detection。

为什么像主会：

- 不是单纯 benchmark，而是结构化表示学习。
- 可以跨 DocQA、TableQA、ChartQA、multi-page QA。
- 可以解释模型错误来源。

最大风险：

- 构图如果只是 parser output，会被认为是 wrapper。
- 必须有 learned graph construction 或 graph refinement 机制。

### 4.2 方向 B：Visual-Structural Self-Verification

一句话：

```text
让模型在回答前生成结构化视觉证据，并把证据渲染回图像中进行自检和修正。
```

核心机制：

- 模型先预测 evidence：bbox、reading order、table cells、chart points、answer spans。
- 系统把这些 evidence 画回页面图像。
- 模型基于 overlay image 判断证据是否支持答案。
- 如果不支持，触发局部重读或修正。

为什么像主会：

- 承接 Visual Self-Refine，但扩展到 document parsing / DocQA。
- 中间过程可视化，容易做诊断。
- 可以自然缓解 hallucination。

最大风险：

- 需要证明不是简单多次 retry。
- 要和 text-only self-refine、CoT、direct MLLM、parser+LLM 比。

### 4.3 方向 C：Cost-Aware Adaptive Reading

一句话：

```text
文档理解不应该所有页面都高分辨率读，也不应该所有问题都走 OCR/parser/RAG/full-vision；模型应按问题动态选择读取路径。
```

核心机制：

- 低成本 skim stage：快速判断问题需要哪些页面/模态。
- routing stage：选择 text-only、vision-only、parser、table specialist、chart specialist、high-res crop、multi-page aggregation。
- verification stage：检查答案是否被 evidence 支撑。
- cost-aware objective：准确率和 token/latency/API cost 一起优化。

为什么像主会：

- 类似 TableDART，但扩展到多页文档解析。
- 直接解决文档解析的成本痛点。
- 容易做 cost-performance Pareto curve。

最大风险：

- 如果 routing 只是规则，会被认为工程系统。
- 需要 learned router、oracle upper bound、simple heuristic baseline、routing error analysis。

### 4.4 方向 D：Parser-in-the-Loop Reasoning

一句话：

```text
把 parser 从固定预处理工具变成可被 MLLM 反向验证和修正的中间模块。
```

核心机制：

- parser 给出 OCR/layout/table/chart 初始结构。
- MLLM 根据问题和答案候选检测 parser inconsistency。
- 对关键区域发起局部 reparse / crop / high-res OCR。
- parser 和 reasoner 形成闭环，而不是单向 pipeline。

为什么像主会：

- 回应传统 pipeline 的 error propagation。
- 比 end-to-end 更可控，比 parser+LLM 更智能。
- 可以分析 parser error 如何影响 reasoning。

最大风险：

- 容易被认为是 agent workflow。
- 必须有明确训练目标和模块消融。

### 4.5 方向 E：Structured Output Decoding

一句话：

```text
把文档解析输出从自然语言答案改成可验证结构：HTML / Markdown AST / JSON graph / SVG / TikZ / layout program。
```

核心机制：

- 设计文档结构语言。
- 模型生成结构化程序。
- 通过 rendering / execution / validation 检查输出。
- 用结构一致性 reward 或 constrained decoding 优化。

为什么像主会：

- 承接 DaVinci 的 structured code 思路。
- 输出可执行、可验证，不只是 open-ended answer。
- 能自然引入 RL 或 verifier。

最大风险：

- 如果只是换输出格式，会被认为 formatting trick。
- 需要证明结构语言带来更强泛化和更少 hallucination。

---

## 5. 我最推荐的 paper 方向

### 5.1 推荐题目

```text
Visual-Structural Self-Verification for Evidence-Grounded Document Parsing
```

也可以换成更短的名字：

```text
DocVerify: Visual-Structural Self-Verification for Document Intelligence
```

### 5.2 核心主张

现有 MLLM 在文档解析和问答中的主要问题，不只是 OCR 不准，也不只是推理不强，而是：

```text
模型缺少一个可视化、可验证、可修正的 evidence intermediate representation。
```

因此它容易：

- 读错文本但不知道自己读错；
- 找错区域但仍然自信回答；
- 把无关 table cell 当成证据；
- 在 chart / figure / caption 间错配；
- 在多页文档中凭语言先验补答案；
- 生成答案但没有 grounded provenance。

本文提出一个方法：让模型在回答前后都显式生成 visual-structural evidence，并通过渲染自检闭环来修正答案。

### 5.3 方法框架

#### Stage 1：Evidence Proposal

输入：

- document pages
- user question
- optional OCR/parser outputs

模型输出候选 evidence：

- page id
- bbox
- OCR span
- table cell coordinate
- chart mark / axis / legend region
- figure-caption relation
- reading order
- answer-supporting relation

输出格式可以是 JSON：

```json
{
  "evidence": [
    {
      "page": 3,
      "type": "table_cell",
      "bbox": [120, 340, 480, 390],
      "text": "Revenue 2024: 12.8M",
      "role": "supporting_fact"
    }
  ],
  "reasoning_type": "table_lookup_with_arithmetic"
}
```

#### Stage 2：Evidence Rendering

把 evidence 画回文档图像：

- 高亮 bbox
- 标出 cell / row / column
- 标出 reading order
- 标出 chart point / legend / axis
- 用颜色区分 supporting / distractor / uncertain evidence

这一步的关键是：**让模型重新看到自己的证据选择**。

#### Stage 3：Visual-Structural Verification

模型基于 overlay image 回答几个 verification questions：

- evidence 是否真的包含答案？
- evidence 与 question entity 是否匹配？
- 是否遗漏了表格行/列？
- 是否引用了错误页面？
- 是否存在多个候选答案？
- 是否需要局部 high-res reread？

输出：

```json
{
  "verification": "fail",
  "error_type": "wrong_table_column",
  "repair_action": "reread_crop",
  "target_region": [100, 300, 520, 430]
}
```

#### Stage 4：Local Repair and Final Answer

如果 verification fail，则触发：

- high-res crop OCR
- table cell re-localization
- chart point re-estimation
- page-level retrieval rerank
- answer regeneration

最终输出：

- answer
- evidence trace
- confidence
- failure category if uncertain

### 5.4 这篇论文的方法创新点

#### 创新 1：Evidence 不是文本解释，而是 visual-structural object

很多 CoT 只是文本推理链。这里的 evidence 是可渲染的结构对象，能被视觉检查。

#### 创新 2：Self-verification 不是让模型“再想想”，而是让模型检查自己选中的证据

区别于 text-only self-refine：

- text-only self-refine 只能检查语言一致性；
- visual-structural self-verification 能检查 bbox、cell、row/column、chart mark、page region 是否真的对。

#### 创新 3：修正是局部的，而不是重新跑整篇文档

如果错误发生在一个 table cell 或 chart point，只重新读取相关 crop。这样可以控制成本。

#### 创新 4：方法能同时服务 parsing、QA、hallucination detection

它不是单个 benchmark trick，而是一个中间机制：

- DocQA：答案必须有 evidence。
- OCR hallucination：没有 evidence 的答案判为 hallucination。
- TableQA：cell evidence 可验证。
- ChartQA：mark / axis evidence 可验证。
- Long-document QA：page-level provenance 可验证。

### 5.5 为什么这比 HalluText 式 benchmark 更强

HalluText 的问题是把重点放在 benchmark 和 mitigation 上，但 reviewer 觉得：

- benchmark 构建不够透明；
- 方法依赖外部 OCR；
- mitigation 的机制和 baseline 不够硬；
- 任务范围偏窄。

DocVerify 这类方法型论文可以避开一部分风险：

- 不只定义 hallucination，而是提出 evidence verification mechanism。
- 不只测模型错，而是修正模型错。
- 不只看 OCR hallucination，而是覆盖 layout、table、chart、multi-page evidence。
- 不只给 final answer，而是给可渲染 evidence trace。

### 5.6 和已有 accepted papers 的关系

这篇可以自然对齐主会中稿信号：

- 像 Visual Self-Refine：有 visual feedback，但扩展到 broader document parsing。
- 像 TableDART：有 adaptive action / repair routing，但不是只做表格。
- 像 DaVinci：有 structured output 和 verifier，但输出不是 TikZ，而是 evidence object。
- 像 DAVE：可以进一步接入 document-specific encoder，但主贡献不依赖新 encoder。
- 像 OCR-Reasoning：能评估 reasoning process，但更进一步让 process grounded 到视觉证据。

---

## 6. 实验设计

### 6.1 任务选择

建议不要只做一个任务。至少覆盖四类：

1. Text-rich DocQA
2. Table QA
3. Chart QA / chart parsing
4. Multi-page long-document QA

可选再加：

- OCR hallucination detection
- evidence localization
- figure-caption reasoning
- financial / scientific report QA

### 6.2 Baselines

必须包含：

- Direct MLLM
- OCR + LLM
- parser + LLM
- RAG + LLM
- MLLM + CoT
- MLLM + text-only self-refine
- MLLM + multi-round retry
- strong document parser + LLM
- high-resolution crop baseline
- oracle evidence upper bound

如果做表格/图表，还要加：

- table-specific parser / model
- chart-specific parser / model
- TableDART-like routing if applicable
- Visual Self-Refine-like chart baseline

### 6.3 Ablation

至少做这些：

- without evidence rendering
- without verification
- without local repair
- text evidence only
- visual bbox only
- parser output only
- learned verifier vs rule-based verifier
- single-pass vs iterative repair
- low-res vs high-res repair
- cost-aware repair threshold

### 6.4 指标

不要只报 accuracy。

建议指标：

- answer accuracy
- evidence precision / recall
- bbox IoU
- table cell accuracy
- page retrieval recall
- hallucination rate
- unsupported answer rate
- repair success rate
- token cost
- latency
- number of model calls
- human preference on evidence faithfulness

### 6.5 关键图表

论文里最好有这些图：

- method overview
- evidence rendering example
- error before / after repair
- accuracy vs cost Pareto curve
- hallucination rate reduction
- performance by evidence type
- repair action distribution
- failure taxonomy

---

## 7. Reviewer 可能怎么攻击

### 攻击 1：这不就是 Visual Self-Refine 扩展到文档吗？

回应策略：

- 强调 Visual Self-Refine 主要针对 chart point localization。
- 本文定义更一般的 evidence object：text span、bbox、cell、chart mark、caption、page。
- 本文有 verification + local repair + provenance，而不是只 refine pixel point。
- 多任务结果证明机制不是 chart-specific。

### 攻击 2：这不就是 parser + LLM pipeline？

回应策略：

- parser 只是 optional proposal，不是固定答案来源。
- 方法会验证和修正 parser output。
- 做 parser error injection，看方法是否能恢复。
- 对比 parser + LLM、parser + CoT、parser + self-refine。

### 攻击 3：这不就是多轮 retry？

回应策略：

- 多轮 retry 没有结构化 evidence 和 visual rendering。
- 本文 repair action 是局部、可解释、可统计的。
- ablation 显示 random retry / text retry 不如 evidence-guided repair。

### 攻击 4：成本太高

回应策略：

- 只对 verification fail 的样本触发 repair。
- 报 cost-performance Pareto。
- 加 cost-aware threshold。
- 展示在相同 token budget 下优于 multi-agent / full high-res reread。

### 攻击 5：evidence 标注成本太高

回应策略：

- 使用 parser/OCR 自动 proposal + 人工验证 subset。
- 部分任务可从已有 DocVQA/TableQA/ChartQA 标注转化。
- 主要方法可以从 weak supervision 学习。
- 展示少量 evidence supervision 也能提升。

---

## 8. 论文结构建议

### Abstract

核心句式：

```text
Current MLLMs often answer document questions without visually verifiable evidence,
leading to hallucinations and brittle reasoning over text, tables, and charts.
We propose Visual-Structural Self-Verification, a framework that requires models
to generate, render, verify, and repair structured evidence before producing final answers.
```

### Introduction

建议结构：

1. 文档解析不是普通 VQA，因为答案依赖细粒度结构证据。
2. 当前 MLLM 的失败不是单纯 OCR 错，而是 answer 与 visual evidence 脱钩。
3. 现有路线的问题：
   - OCR/parser pipeline 有 error propagation。
   - CoT/self-refine 主要在文本空间自洽。
   - agent/RAG 系统缺少可验证中间证据。
4. 本文提出 visual-structural evidence object 和 self-verification loop。
5. 贡献总结：method、training/evaluation、experiments、analysis。

### Method

小节：

1. Evidence Object Definition
2. Evidence Proposal
3. Evidence Rendering
4. Visual-Structural Verification
5. Local Repair Policy
6. Training Objective / Prompting Setup
7. Complexity Analysis

### Experiments

小节：

1. Tasks and Datasets
2. Baselines
3. Main Results
4. Ablations
5. Cost-Performance Tradeoff
6. Error Analysis
7. Human Evaluation

### Limitations

要主动承认：

- evidence rendering 增加系统复杂度；
- 对 parser proposal 有一定依赖；
- 极复杂图表/低质量扫描仍然困难；
- evidence annotation 的规模化仍需进一步研究。

---

## 9. 另两个备选 paper 方案

### 9.1 Adaptive Modality Routing for Long Document Intelligence

核心：

把 TableDART 的 dynamic routing 扩展到多页文档。

贡献：

- learned router 选择 text / image / parser / table / chart / high-res crop / retrieval。
- 目标函数同时优化 accuracy 和 cost。
- 输出 routing trace 和 evidence trace。

适合场景：

- 长文档 QA
- 企业报告
- scientific paper QA
- financial document QA

最大风险：

- 容易被认为是 agent workflow。
- 必须有 learned router 和强 efficiency 实验。

### 9.2 Learning Document Layout Programs

核心：

把文档页面解析成可执行 layout program，而不是自然语言描述。

贡献：

- 定义 document layout DSL。
- 模型生成 program。
- program 可以 render 回页面并验证。
- 用 rendering similarity + structural consistency 做训练/评估。

适合场景：

- PDF parsing
- scanned document reconstruction
- table / figure / caption linking
- HTML / Markdown restoration

最大风险：

- 如果 DSL 设计太工程化，会被认为 niche。
- 必须证明 program representation 能提升 downstream QA / retrieval。

---

## 10. 最终建议

如果你想冲主会，我最推荐：

```text
Visual-Structural Self-Verification for Evidence-Grounded Document Parsing
```

因为它兼具：

- 方法创新：visual-structural evidence + verification + local repair。
- 任务广度：DocQA、TableQA、ChartQA、long-document QA、hallucination。
- 可解释性：evidence trace 可视化。
- 实验空间：可以做丰富 ablation 和 cost trade-off。
- 审稿防御力：能回应 pipeline、prompt trick、agent 拼装、benchmark-only 的质疑。

这篇的关键不是造一个更大的数据集，而是提出一个 reviewer 能记住的机制：

```text
模型必须先把答案锚定到可渲染的视觉结构证据上，
再通过视觉自检发现并修正错误。
```

如果后续继续推进，我建议下一步不是马上收数据，而是先写一个 2 页 mini proposal：

1. problem definition
2. method diagram
3. evidence object schema
4. 3 个代表性 case
5. baseline table
6. expected ablation

这样可以尽早判断它到底像主会方法论文，还是会滑回工程系统。
