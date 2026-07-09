# ICLR 2026 主会文档解析方向复盘报告

整理时间：2026-07-07  
范围：ICLR 2026 主会中与 document parsing / document intelligence / OCR / table / chart / diagram / long-document retrieval / DocQA / web-document encoder 相关的代表性论文  
说明：这里的“track”指主题方向，不是 ICLR 官方 track 名称。本文关注主会审稿口味，不局限于 benchmark paper。

---

## 1. 总结先行

这一轮重新看下来，ICLR 主会对“文档解析方向”的接受标准不是“只要做了一个 benchmark 就行”，也不是“只要工程系统效果好就行”。更准确地说，主会愿意接受以下几类工作：

- **文档/网页/表格/图表场景下的 representation learning**，例如专门的 vision encoder。
- **结构化对象的解析机制**，例如 chart parsing、scientific diagram parsing、flowchart synthesis。
- **面向结构化信息的高效推理系统**，例如 table modality routing、adaptive reasoning。
- **能改变训练或评测分布的大规模资源**，例如真实视觉表格、infographic chart、flowchart generator。
- **有诊断价值的 benchmark**，但前提是它能揭示新能力或新失败模式，而不是只做排行榜。
- **retrieval / RAG / agent / compression 系统**，但门槛更高：必须证明不是简单拼装，也必须把效率、baseline、泛化讲硬。

一句话：**主会看重的是“文档解析问题背后的可泛化研究命题”，而不是文档解析这个应用标签本身。**

中稿论文通常能把自己讲成一个更大的研究问题：

- DAVE：通用视觉编码器缺少文档和网页所需的结构/空间特征。
- TableDART：表格理解不应该静态融合所有模态，而应按实例动态路由。
- Visual Self-Refine：视觉密集解析需要 pixel-level self-verification，而不是只做文本式 self-correction。
- DaVinci：scientific diagram parsing 需要学习 visual-structural syntax，并用 RL reward 对齐结构和代码。
- FlowGen：流程图理解需要可控的结构复杂度和渲染风格，而不是固定小数据集。
- TABLET：视觉表格训练存在 real rendering 与 synthetic rendering 的 train-test mismatch。

拒稿论文往往不是方向错，而是没有把“主会级贡献”立住：

- 像 prompt / formatting trick。
- 像 RAG / agent 组件拼装。
- 像 dataset engineering 但缺少可验证 insight。
- 像 benchmark 但没有足够数据可信度。
- 像 compression / efficiency 方法但没有赢过最简单强 baseline。
- 像 end-to-end 系统但没有证明成本、泛化和现代 parser baseline。

---

## 2. 本轮调研样本

### 2.1 深挖的中稿论文

| 论文 | 类型 | 结果 | OpenReview |
|---|---|---:|---|
| DAVE: A VLM Vision Encoder for Document Understanding and Web Agents | 文档/web vision encoder | Poster | https://openreview.net/forum?id=kgk0NqjsoW |
| TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding | 动态多模态路由系统 | Poster | https://openreview.net/forum?id=4aZTiLH3fm |
| TableMaster: A Recipe to Advance Table Understanding with Language Models | 表格理解 recipe / system | Poster | https://openreview.net/forum?id=YyPZPrPjQD |
| Beyond Text-Only: Towards Multimodal Table Retrieval in Open-World | 视觉表格检索 / retrieval benchmark | Poster | https://openreview.net/forum?id=4QPgqdQmYn |
| TABLET: A Large-Scale Dataset for Robust Visual Table Understanding | 视觉表格训练资源 | Poster | https://openreview.net/forum?id=5UbeQDlYDj |
| ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation | infographic chart 数据资源 | Poster | https://openreview.net/forum?id=P4lFbvZ4HH |
| Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing | chart parsing 方法 | Poster | https://openreview.net/forum?id=RI0oNr1b0y |
| DaVinci: Reinforcing Visual-Structural Syntax in MLLMs for Generalized Scientific Diagram Parsing | scientific diagram parsing / SFT+RL | Poster | https://openreview.net/forum?id=OAXECnLxuk |
| FlowGen: Synthesizing Diverse Flowcharts to Enhance and Benchmark MLLM Reasoning | controllable flowchart synthesis | Poster | https://openreview.net/forum?id=uimrBBfDCH |
| OCR-Reasoning Benchmark | text-rich image reasoning benchmark | Poster | https://openreview.net/forum?id=aH7eyx64pC |

### 2.2 深挖的拒稿论文

| 论文 | 类型 | 结果 | OpenReview |
|---|---|---:|---|
| HalluText | OCR hallucination benchmark + mitigation | Reject | https://openreview.net/forum?id=LRnt6foJ3q |
| End-to-End Document Understanding via Chain-of-Reading | end-to-end document QA / CoT | Reject | https://openreview.net/forum?id=6YXMyPrDEN |
| Structured Attention Matters to MLLMs in Document Understanding | structured input / attention analysis | Reject | https://openreview.net/forum?id=3OnJAvuxd3 |
| GDI-Bench | general document intelligence benchmark + fine-tuning | Reject | https://openreview.net/forum?id=4l8QRqYzH9 |
| MDocAgent | multi-modal multi-agent DocQA / RAG | Reject | https://openreview.net/forum?id=05SHW9ai9e |
| CoTabBench | weakly structured table QA benchmark + model | Reject | https://openreview.net/forum?id=wcInjlUp8V |
| ChartNexus | multi-chart reasoning benchmark | Reject | https://openreview.net/forum?id=xg0fmtqh8d |
| VisR-Bench | multilingual long-document visual retrieval/RAG benchmark | Reject | https://openreview.net/forum?id=7iFZ6uzILL |
| DocPruner | visual document retrieval compression | Reject | https://openreview.net/forum?id=mEMGL1fLOO |

---

## 3. 主会接收的几种贡献形态

### 3.1 Representation / Encoder 型

代表论文：DAVE  
核心问题：通用 VLM 的 vision encoder 对 document / web / UI 场景不够敏感，缺少结构、空间、OCR-heavy 特征。

DAVE 的贡献不是做一个新 benchmark，而是提出一个专门面向文档和网页任务的 vision encoder。它结合 self-supervised pretraining、supervised autoregressive pretraining、model merging、ensemble training，并在 document understanding、VQA、web localization、agent benchmark 等任务上验证。

为什么能中：

- 方向站得住：文档和网页确实对视觉编码器有特殊要求。
- 实验面足够宽：不是只在一个 DocVQA 数据集上上涨点。
- reviewer 虽然质疑 novelty，但承认组合非平凡且结果强。
- rebuttal 补了 cross-domain evaluation、visual encoder comparison、SigLIP2 依赖、variance 等问题。

保留问题：

- 方法是已有训练技巧的组合，novelty 不是特别强。
- 在 general VQA 上有一些 regression。
- RICO-SCA 上低于 SigLIP2 的问题没有完全解释。

主会信号：

**文档解析方向可以发 representation learning，但必须证明它不是“换一个数据域微调”，而是捕捉到了文档/网页场景中特有的视觉结构。**

---

### 3.2 Dynamic Routing / Adaptive System 型

代表论文：TableDART  
核心问题：表格可以看成 text，也可以看成 image，还可以 multimodal fusion；但静态地把所有模态都塞进去，会带来冗余、冲突和高成本。

TableDART 用一个 2.59M 参数的 gating network，在 Text-only、Image-only、Fusion 三条路径之间动态选择。Fusion 阶段用 LLM agent 仲裁 unimodal 输出或合成新答案。

为什么能中：

- 问题定义有机制感：不是“表格理解难”，而是“不同表格/问题需要不同模态路径”。
- 方法轻量，和 full MLLM fine-tuning 形成对比。
- 七个 benchmark 上有比较强的结果。
- rebuttal 加了 TabPedia、SynTab-LLaVA、Qwen2.5-VL 等强 baseline，补了 ablation 和 efficiency analysis。
- AC 认为两个低分 review 有误解，rebuttal 成功澄清。

保留问题：

- Fusion agent 仍偏黑盒。
- 使用 Gemini 2.0 Flash 带来 reproducibility/accessibility 问题。
- gating supervision 的 correctness vector 构造仍可进一步审计。

主会信号：

**系统型工作可以中，但要把系统抽象成一个清晰机制：动态路由、冲突消解、效率收益、跨模型可复用。**

---

### 3.3 Recipe / Problem Decomposition 型

代表论文：TableMaster  
核心问题：表格理解不是单一能力，而是 data localization、table semantics、numerical reasoning、symbolic reasoning 多个瓶颈叠加。

TableMaster 把问题拆成四类挑战，并设计 table-of-focus、verbalization、adaptive textual/symbolic reasoning。

为什么能中：

- 问题拆解清楚，reviewer 容易理解每个模块解决什么。
- 虽然组件多来自已有思路，但组合成一个 cohesive pipeline。
- 实验覆盖多个 benchmark 和模型，包括 proprietary 与 open-source。
- rebuttal 补了 ablation、latency、不同模型规模等问题，AC 判断 reviewer 分数整体会上调。

保留问题：

- 单个模块 novelty 有限。
- 更像 practical recipe，而不是算法突破。

主会信号：

**如果技术不是单点突破，可以用“清晰问题分解 + 系统实证”支撑主会贡献。**

---

### 3.4 Retrieval / RAG 型

代表中稿：Beyond Text-Only / TaR-ViR  
代表拒稿：VisR-Bench、DocPruner

#### Beyond Text-Only 为什么能中

这篇把 open-domain table retrieval 从 text-only 改成 multimodal table retrieval，主张 table image 可以保留结构、空间和嵌入图片信息，避免 text serialization 丢结构。

中稿原因：

- 有清晰 paradigm shift：table retrieval 不一定要把表格 flatten 成文本。
- 建了 TaR-ViR，约 2M Wikipedia table screenshots，并对齐 natural-language queries。
- 对比 text retrievers 和 multimodal retrievers，展示视觉检索在 recall 和大规模检索上的潜力。
- rebuttal 补了 resolution ablation、HTML/OCR/text format comparison、missing title experiments、更大 Qwen2-VL 结果。

剩余风险：

- 主要来自 Wikipedia，真实企业表格、扫描文档、复杂 spreadsheet 覆盖不足。
- semi-automated annotation 有 80% precision，训练噪声需要接受 scale-quality trade-off。
- RAG 下游提升有限，text-only LLM 仍可能更强。

#### VisR-Bench 为什么被拒

VisR-Bench 关注 multilingual long-document visual retrieval/RAG，本身方向很现实。但最终被拒，说明 retrieval benchmark 的主会门槛不低。

主要问题：

- 初始版本缺少 o3、Gemini 2.5 等 reasoning-capable MLLM。
- 过度依赖 Top-1 retrieval metric，不能充分代表真实 RAG。
- 合成 QA 的人工验证规模小。
- LLM-based heuristics 带来的偏差缺少 robust error analysis。
- multilingual 覆盖排除中文等 logographic language，削弱多语言主张。

rebuttal 结果：

- 作者补了 o3、Gemini 2.5、人类验证、multilingual statistics。
- reviewer 承认透明度变好，但 AC 认为 novelty、cross-language fairness、error analysis 仍不足。

#### DocPruner 为什么被拒

DocPruner 关注 multi-vector visual document retrieval 的 storage overhead，这个问题很重要。它提出 adaptive patch-level embedding pruning，可以减少 50-60% 存储。

被拒的核心不是问题不重要，而是实验没有证明关键机制：

- attention-based pruning 被认为是常见方法。
- 没有证明 adaptive threshold 明显优于 simple fixed-ratio attention pruning。
- 图表缺失数据点、baseline 曲线不清楚。
- rebuttal 没有补出完整 trade-off curve 和清晰 baseline。

主会信号：

**retrieval/RAG/efficiency 工作必须有完整链路证据：retrieval 指标、downstream QA、成本曲线、强 baseline、误差来源。单独优化一个指标通常不够。**

---

### 3.5 Structured Parsing / Visual Feedback 型

代表论文：Visual Self-Refine  
核心问题：chart parsing 错误很多来自视觉定位，而文本 self-correction 对视觉感知帮助有限。

Visual Self-Refine 让模型生成 pixel-level localization outputs，把这些输出可视化后再反馈给模型进行修正。它不是让模型只在文本里反思，而是把中间视觉证据显式渲染出来。

为什么能中：

- 有一个清楚的机制：pixel-guided self-verification。
- chart parsing 场景非常适合这个机制。
- rebuttal 补了 Qwen3、InternVL、Gemini 2.5 等新模型对比。
- AC 接受作者说法：这不是临时 workaround，而是 system-2-like visual verification。

保留问题：

- 方法比较 task-specific。
- 额外 refine calls 增加推理成本。
- 在部分旧数据集上提升不稳定。

主会信号：

**文档解析里的“中间视觉证据”是一个重要方向。主会更喜欢可操作的中间机制，而不是只说模型需要更好 reasoning。**

---

### 3.6 Scientific Diagram / Structured Code 型

代表论文：DaVinci  
核心问题：scientific diagram parsing 不是普通 OCR，也不是普通 image captioning，而是要把 raster diagram 解析成结构化 TikZ code。

DaVinci 采用两阶段：

- SFT 学习 visual primitives。
- RL 学习 structural relationships。

它还设计了 TikZ30K 数据、优化 drawing order、comments placement，以及包含 visual fidelity、structural consistency、code correctness 的 hybrid reward。

为什么能中：

- 任务非常具体且有研究含量：从图像到结构化可编辑代码。
- 方法不是单纯 prompting，而是 SFT + RL + reward design。
- 数据顺序、注释、结构 reward 等细节有实验支撑。
- rebuttal 解决了 dataset transparency、size、evaluation bias、human evaluation、technical claims 等担忧。
- 有 reviewer 从 4 提升到 6，AC 正向推荐。

保留问题：

- 自动指标与 reward 有重叠，可能有 metric hacking 风险。
- license 和 human evaluation 是初审关注点。

主会信号：

**文档解析如果能落到“结构化输出语言”上，例如 HTML、LaTeX、TikZ、SVG、Markdown AST、layout graph，会比普通 QA 更容易形成主会级技术问题。**

---

### 3.7 Controllable Synthetic Data / Generator 型

代表论文：FlowGen  
核心问题：流程图理解缺少可控结构复杂度和渲染风格，现有数据集无法系统评估 MLLM 的结构推理能力。

FlowGen 是一个可控 flowchart synthesizer，支持 Mermaid、Graphviz、PlantUML、Diagrams 等 renderer，可以控制 graph size、branching、nested subgraphs、split/merge arrows 等属性。

为什么能中：

- generator 本身有清楚控制变量，不只是随机合成数据。
- 同时用于 training 和 benchmarking。
- 实验覆盖 open-source 和 proprietary MLLM。
- 有 exact 和 relaxed metrics。
- rebuttal 加了 end-to-end fine-tuning、cross-domain checks，证明不是只会学 renderer bias。
- 还能迁移到 FlowVQA、FlowLearn 和 real-world hand-drawn data。

保留问题：

- 合成图是否覆盖真实流程图噪声仍是长期问题。
- LLM 生成 label 可能带来语言风格偏差。

主会信号：

**合成数据能中，但要做到“可控生成 + 真实迁移 + 复杂度分析 + 下游训练收益”。只说 synthetic scalable 不够。**

---

### 3.8 Large Dataset / Resource 型

代表论文：TABLET、ChartGalaxy  
核心问题：现有表格/图表资源与真实视觉分布不匹配。

#### TABLET

TABLET 构建了 4M visual table understanding examples，基于 2M unique tables，其中 88% 保留 original visualizations，并提供 image-HTML pairs、metadata、provenance。它还引入 VisualTableQA，强调视觉感知与表格理解的联合能力。

中稿原因：

- 抓住 train-test mismatch：synthetic rendering 和 real visual table 差距大。
- 数据不仅能评测，还能训练模型。
- rebuttal 加了 Gemma 3-4B cross-model validation、visual complexity analysis、human-annotated visual QA test set。
- reviewer 认可它是 substantial engineering and resource contribution。

剩余问题：

- 主要来源仍偏 Wikipedia。
- 方法 novelty 有限。
- exact visual factors 只做 aggregate analysis。

#### ChartGalaxy

ChartGalaxy 是 million-scale infographic chart dataset，包含 1,701,356 programmatic charts 和 61,833 real infographic charts，覆盖 75 chart types、440 variations、68 layout templates。

中稿原因：

- 数据规模和 diversity 强。
- 同时服务 infographic understanding、code generation、example-based generation。
- rebuttal 解决 template、QA quality、metric reliability、URL correctness 等问题。
- 最终 conditional acceptance，要求补版权和 terms 合规证据。

主会信号：

**资源型论文能中，但要证明它改变了训练/评测分布，并且有多个可验证用例。数据来源、license、release policy 也会直接影响最终决定。**

---

### 3.9 Benchmark / Evaluation 型

代表中稿：OCR-Reasoning  
代表拒稿：HalluText、GDI-Bench、CoTabBench、ChartNexus、VisR-Bench

benchmark 不是不能中，但 accepted 和 rejected 的区别很清楚。

OCR-Reasoning 能中，因为：

- 它评估 text-rich image reasoning，不只是 OCR recognition。
- 1,069 个样本虽然不大，但人工标注质量高。
- 同时提供 final answer 和 step-by-step reasoning trace。
- 结果能诊断模型是读错、推理错还是过程错。
- rebuttal 后 final rating 约为 8 / 8 / 6 / 6。

HalluText / GDI-Bench / CoTabBench / ChartNexus / VisR-Bench 被拒，原因各不相同：

- HalluText：benchmark 构建细节和 mitigation 方法证据不够。
- GDI-Bench：taxonomy、annotation audit、benchmark-method linkage 不够稳。
- CoTabBench：数据构建、train/eval overlap、LLM-generated QA human validation 不清楚，且无 rebuttal。
- ChartNexus：multi-chart setting 有意义，但 novelty / insight 不足。
- VisR-Bench：方向真实，但 Top-1 retrieval、合成 QA 验证、多语言公平性没有打穿。

主会信号：

**benchmark paper 要中，不能只是“我们发现模型差”。它要定义一个新能力、提供可信数据、并产出非显然诊断 insight。**

---

## 4. 拒稿论文的非 benchmark 教训

### 4.1 End-to-End Document Understanding via Chain-of-Reading

贡献类型：end-to-end document QA / multimodal chain-of-thought  
结果：Reject

这篇提出 CoR，让模型直接吃 PDF pages，先定位 evidence，再 OCR，再推理，目标是替代传统 OCR + LLM pipeline。

拒稿原因：

- 泛化性不足：训练和实验主要集中 academic/government reports。
- locate-then-reason 的 novelty 被认为接近 DocReact、agentic workflow 或 standard CoT。
- 缺少 MinerU 等现代 parser baseline。
- 缺少 few-shot prompting、non-Qwen backbone、general VQA / broader document tasks。
- 长 reasoning trace 的 token/latency 成本没有给出硬数字。
- rebuttal 里很多关键结果是“final version 会补”，而不是当前已经补出。

主会教训：

**想证明 end-to-end 优于 pipeline，就必须直接打强 pipeline baseline，并给出成本收益表。**

---

### 4.2 Structured Attention Matters

贡献类型：structured input / attention analysis  
结果：Reject

这篇认为 LaTeX-like structured text 能保留文档层级和空间关系，从而提升 MLLM 文档 QA。

拒稿原因：

- “纯文本丢结构”被认为是已有共识。
- LaTeX 格式输入像 prompt / formatting trick。
- structured text generator 细节不足。
- 没有 HTML / XML / Markdown 等强格式 baseline。
- attention visualization 没有 causal evidence。
- 没有 formal rebuttal。

主会教训：

**training-free formatting 方法要特别小心。除非你能证明机制、对照、效率、泛化都很强，否则很容易被看成 prompt engineering。**

---

### 4.3 MDocAgent

贡献类型：multi-agent DocQA / RAG  
结果：Reject

这篇用 General、Critical、Text、Image、Summarization 五个 agent 组合文档 QA。

拒稿原因：

- novelty 弱，像已有 retriever + prompting + RAG + agent orchestration。
- 过度依赖 GPT-4o。
- 没有充分 generalization testing。
- 缺少 systematic error analysis。
- 没有 time / memory / token efficiency。
- 没有 rebuttal。

主会教训：

**agent 系统在主会很难靠“多个 agent 协作”本身成立。必须证明每个 agent 的不可替代性、相对 strong RAG 的增益，以及额外成本合理。**

---

### 4.4 DocPruner

贡献类型：visual document retrieval efficiency / compression  
结果：Reject

这篇想减少 multi-vector VDR 系统的 patch-level embedding 存储成本。

拒稿原因：

- 方向重要，但 attention-based pruning 不新。
- adaptive threshold 没有证明比 fixed-ratio attention pruning 更好。
- 结果图缺失 baseline data points，曲线不清楚。
- rebuttal 没有补完整 trade-off curve。

主会教训：

**efficiency paper 的核心不是“我省了多少”，而是“我的复杂机制为什么比简单机制必要”。**

---

## 5. Rebuttal 如何改变命运

### 5.1 成功 rebuttal 的共同点

成功案例：

- DAVE
- TableDART
- TableMaster
- TABLET
- Beyond Text-Only
- Visual Self-Refine
- DaVinci
- FlowGen

共同点：

- 补了 reviewer 点名要的强 baseline。
- 补了新模型，不只解释为什么没做。
- 补了 ablation、latency、efficiency 或 trade-off。
- 补了 cross-model / cross-domain generalization。
- 把 reviewer 的误解转化为更清晰的 problem framing。
- 让 AC 能写出“concerns are addressed / substantially addressed”。

### 5.2 失败 rebuttal 的共同点

失败或不充分案例：

- Chain-of-Reading
- HalluText
- GDI-Bench
- ChartNexus
- VisR-Bench
- DocPruner

共同点：

- 核心实验仍是承诺，而不是已完成。
- 补充内容没有击中最致命问题。
- reviewer 关心的是 baseline，但作者只解释设定。
- reviewer 关心的是 reliability，但作者只补少量样本验证。
- reviewer 关心的是 novelty/insight，但作者只证明任务更难。
- reviewer 关心的是 simple baseline，但作者没有直接对比。

### 5.3 没有 rebuttal 基本很危险

典型案例：

- Structured Attention Matters
- MDocAgent
- CoTabBench

这些论文的问题并非完全不可救，但没有 rebuttal 导致关键质疑全部保留。对文档解析这种容易被质疑“工程拼装 / 数据构建不透明”的方向来说，rebuttal 几乎是第二次投稿。

---

## 6. 主会审稿口味总结

### 6.1 主会喜欢什么

#### 1. 把文档解析问题抽象成一般 ML 问题

好例子：

- DAVE：domain-specific representation learning。
- Visual Self-Refine：visual self-verification。
- TableDART：adaptive multimodal routing。
- DaVinci：structured syntax learning with RL。
- FlowGen：controllable synthetic generation for structural reasoning。

这些都不只是“我做文档应用”，而是一个更一般的研究命题。

#### 2. 有清晰机制，而不是只有 pipeline

主会不排斥系统，但系统里必须有一个 reviewer 能记住的机制：

- routing
- visual feedback
- structural reward
- model merging
- modality-specific representation
- controllable generation
- adaptive reasoning
- evidence localization

#### 3. 实验能排除 obvious alternative

accepted paper 的 rebuttal 经常补：

- Qwen / InternVL / Gemini / o3 等新强模型。
- simple baseline。
- stronger parser。
- different backbone。
- resolution ablation。
- format comparison。
- latency / token / storage cost。
- human validation。
- cross-domain transfer。

#### 4. 数据资源要能训练，也要能诊断

TABLET、ChartGalaxy、FlowGen 都不只是“我收了数据”，而是能：

- 训练模型。
- 评估模型。
- 控制变量。
- 分析失败模式。
- 支持多个下游任务。

### 6.2 主会不喜欢什么

#### 1. 只像工程拼装

典型风险：

- RAG + agent + prompt。
- parser + LLM。
- OCR + LLM。
- formatting + MLLM。
- attention pruning + threshold。

不是不能做，但必须证明组合后产生了非显然收益。

#### 2. 只有大规模，没有研究洞察

大数据集会被问：

- 为什么这个分布重要？
- 和现有数据差异在哪里？
- 是否带来新能力？
- 是否可训练？
- 是否可诊断？
- 是否有真实迁移？
- license 是否安全？

#### 3. 只证明模型差，没有解释模型为什么差

ChartNexus 和 VisR-Bench 都说明：benchmark 不能只说“当前模型表现不好”。必须给出新的、细粒度、可操作的 insight。

#### 4. 缺强 baseline

文档解析方向 reviewer 很自然会想到：

- MinerU / MinerU2.5
- modern OCR / parser
- OCR + LLM
- parser + LLM
- direct MLLM
- high-resolution MLLM
- Qwen / InternVL / Gemini / GPT / Claude
- strong RAG
- simple fixed-ratio / heuristic baseline
- HTML / XML / Markdown / LaTeX format baseline

缺一个关键 baseline，可能直接导致 reject。

#### 5. 不报成本

文档解析天然高成本：

- 多页图像
- 高分辨率输入
- OCR / layout parser
- retrieval index
- multi-agent
- multi-step reasoning
- visual self-refine
- multi-vector embeddings

所以 token、latency、memory、storage、throughput 不是附属指标，而是主会审稿的一部分。

---

## 7. 如果我们要做文档解析方向，怎么定位更像主会论文

### 7.1 不要只说“我做一个 benchmark”

更好的说法是：

```text
我们发现现有 MLLM 在真实文档解析中的某个结构化能力缺失。
这个能力无法通过现有 OCR / DocVQA / parser benchmark 被隔离评估。
我们提出一个新的任务定义、结构化输出形式或训练机制，
并通过数据、模型、评测和误差分解证明它是一个可泛化的研究问题。
```

### 7.2 选一个清楚的贡献身份

可以考虑以下几种主会定位：

#### A. Representation paper

目标：学习更适合 document / table / chart / UI 的 visual encoder。  
需要：多任务、多域、多 backbone、大量 ablation、general VQA regression 分析。

#### B. Parsing mechanism paper

目标：提出新的结构化解析机制，例如 visual feedback、layout graph decoding、parser-in-the-loop verification。  
需要：比 prompt / parser pipeline 更强的机制证据。

#### C. Structured output / code generation paper

目标：把文档、图表、diagram 解析成 HTML / LaTeX / TikZ / SVG / JSON graph / Markdown AST。  
需要：syntax correctness、rendering fidelity、human eval、reward/metric 防 hacking。

#### D. Adaptive system paper

目标：根据文档和 query 自动选择 OCR、vision、parser、retrieval、symbolic reasoning 路径。  
需要：routing rationale、cost-performance trade-off、simple routing baseline、error attribution。

#### E. Data/resource paper

目标：构建改变训练分布的数据资源，而不是只做 test set。  
需要：provenance、license、quality control、training utility、cross-domain transfer。

#### F. Retrieval/RAG paper

目标：解决长文档、多页、多模态证据定位。  
需要：retrieval + downstream QA 双重评测，Top-k 而不只是 Top-1，真实成本和错误传播。

### 7.3 最推荐的方向

如果结合你前面关注 HalluText 和文档解析，我更建议不要把论文写成单纯 benchmark，而是写成：

```text
Evidence-grounded Document Parsing and Reasoning:
从多页真实文档中生成结构化 evidence graph，
并用该 graph 支撑 OCR、layout、table、chart、QA、hallucination 诊断。
```

这个定位比“做一个 OCR hallucination benchmark”更主会：

- 有结构化输出：evidence graph / layout graph / reading graph。
- 有中间监督：bbox、cell、span、page、relation、source evidence。
- 有方法空间：parser-in-the-loop、visual feedback、adaptive routing、verification。
- 有评测空间：retrieval、parsing、reasoning、hallucination、cost。
- 可以训练模型，也可以评估模型。

### 7.4 一个更像主会的论文题目形态

可以朝这些方向想：

- `Learning Evidence Graphs for Faithful Multimodal Document Reasoning`
- `Parser-in-the-Loop Verification for Long Document Understanding`
- `Adaptive Modality Routing for Evidence-Grounded Document QA`
- `From Pixels to Provenance: Structured Evidence Traces for Document Intelligence`
- `Visual-Structural Self-Verification for Document Parsing`

这些题目都比 `A Benchmark for Document Parsing` 更像主会，因为它们突出机制和研究问题。

---

## 8. 投稿前检查表

### 8.1 贡献定位

- [ ] 这篇是 representation / parsing mechanism / adaptive system / retrieval / resource / benchmark 中的哪一种？
- [ ] 贡献能否用一句 general ML problem 表达？
- [ ] 是否避免被看成 prompt trick / RAG patchwork / parser wrapper？
- [ ] 是否有一个 reviewer 能记住的核心机制？

### 8.2 实验设计

- [ ] 是否包含最新强 MLLM？
- [ ] 是否包含开源和闭源模型？
- [ ] 是否包含 obvious simple baseline？
- [ ] 是否包含 parser / OCR / RAG pipeline baseline？
- [ ] 是否包含 ablation？
- [ ] 是否包含 cross-domain / cross-model / cross-task generalization？
- [ ] 是否包含 efficiency cost？

### 8.3 数据与评测

- [ ] 是否有 provenance / license / release policy？
- [ ] 是否有 human validation？
- [ ] 是否有 train-test leakage check？
- [ ] 是否有 real-world subset？
- [ ] 是否能支持 training，而不只是 testing？
- [ ] 是否有细粒度 error analysis？
- [ ] 是否能产出非显然 insight？

### 8.4 Rebuttal 准备

投稿前最好预留这些补实验：

- [ ] 一个最新 MLLM baseline。
- [ ] 一个现代 parser baseline。
- [ ] 一个 simple heuristic baseline。
- [ ] 一个 latency / token / storage table。
- [ ] 一个 human validation table。
- [ ] 一个 cross-domain transfer result。
- [ ] 一个 failure analysis figure。
- [ ] 一个数据 license / ethics appendix。

---

## 9. 最终判断

ICLR 主会的文档解析方向不是没有机会，反而机会很大，因为文档、表格、图表、网页、流程图、科学图都在暴露当前 MLLM 的结构化视觉理解短板。

但主会不太会因为“文档解析是个重要应用”就接受。更稳的路径是把工作讲成：

```text
一个结构化多模态学习问题，
一个可复用的机制，
一个能训练和诊断模型的数据资源，
或者一个能改变现有模型失败模式理解的评测框架。
```

如果你的目标是做这个方向的主会论文，我建议核心策略是：

1. 先定义一个比 benchmark 更大的能力缺口。
2. 给这个能力设计结构化中间表示。
3. 同时提供方法、数据和诊断评测。
4. 用强 baseline 和成本分析证明它不是工程拼装。
5. 在 rebuttal 前准备好补实验，而不是等 reviewer 问了再想。

最重要的一句话：

**不要让论文看起来像“文档数据集 + 一堆模型评测”。要让它看起来像“文档场景暴露了 MLLM 的一个基础结构化能力缺口，而我们给出了可验证、可训练、可推广的解决路径”。**

