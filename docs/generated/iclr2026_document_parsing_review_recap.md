# ICLR 2026 文档解析方向 OpenReview 复盘报告

整理时间：2026-07-07  
范围：ICLR 2026 OpenReview 中与文档解析、OCR、DocVQA、表格理解、图表理解、长文档检索相关的代表性中稿与拒稿论文  
用途：为后续设计自己的 benchmark / dataset / evaluation paper 提供参考

---

## 1. 一句话结论

ICLR 2026 这一批文档解析相关论文里，**能中稿的 benchmark / dataset 文章通常不是只“收了一个数据集”，而是让审稿人相信三件事：问题定义可靠、数据构建可信、评测能产生新的诊断洞察**。

相反，很多拒稿论文的问题并不一定是方向不重要，而是卡在以下几类硬伤：

- 数据构建细节不够透明，别人无法相信 benchmark 真的可靠。
- QA / 标注 / 合成流程缺少人工验证或一致性分析。
- 对照实验不够强，缺少当下最强模型、强 parser、强 RAG、强 prompt baseline。
- benchmark 只是证明“模型会掉点”，但没有解释为什么掉、掉在哪里、如何诊断。
- 方法像 prompt trick / agent 拼装 / formatting trick / pruning trick，缺少超越 obvious baseline 的证据。
- rebuttal 只承诺补实验，而不是已经补出关键证据。

如果你也想做 benchmark，最重要的不是先追求“大”，而是先把 **benchmark trust chain** 做扎实：来源、清洗、标注、质检、难度定义、泄漏控制、评测协议、误差分析、复现资产，每一环都要能经得起追问。

---

## 2. 检索与分析范围

本报告基于 OpenReview 上 ICLR 2026 Conference 页面中可见的接收与拒稿论文，围绕以下关键词筛选：

- OCR
- DocVQA
- document understanding
- document parsing
- layout
- PDF
- table
- chart
- visual retrieval
- multimodal long document
- text-rich image reasoning

主要参考页面：

- ICLR 2026 Conference 页面：https://openreview.net/group?id=ICLR.cc/2026/Conference
- OpenReview 论文详情页与 meta-review / official reviews / author response

说明：

这不是一次完整数据库级别的全量统计，而是围绕文档解析方向做的代表性深挖。重点不在“列完所有论文”，而在复盘审稿逻辑：为什么类似方向有的能中，有的会被拒。

本报告正文逐篇复盘 14 篇论文，并为这 14 篇补充了 rebuttal 结果。参考列表中的 TABLET 作为后续可继续深挖的补充候选保留；本轮没有抓到足够完整的 rebuttal 讨论，因此不把它纳入逐篇结论。

---

## 3. 中稿论文复盘

### 3.1 OCR-Reasoning Benchmark

论文：OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning  
OpenReview：https://openreview.net/forum?id=aH7eyx64pC  
结果：Accept Poster  
评分：8 / 8 / 6 / 4

#### 核心贡献

这篇文章提出一个面向复杂 text-rich image reasoning 的 OCR reasoning benchmark。它不是只考 OCR 识别，而是考模型能否基于图像中的文字进行多步推理。

数据规模约 1,069 个高质量人工标注样本，覆盖 6 类核心推理能力和 18 个任务。一个重要设计是同时提供：

- final answer
- step-by-step reasoning trace

这让 benchmark 不只是看最终对错，还可以诊断模型推理过程中的错误。

#### 为什么能中

1. **问题定义清楚**

   审稿人能理解它要评估的不是普通 OCR，也不是泛泛的 VQA，而是 text-rich image 上的复杂推理能力。

2. **标注设计有诊断价值**

   step-by-step reasoning trace 是关键。它让 benchmark 从“排行榜”变成了“诊断工具”。

3. **人工标注增强可信度**

   对 benchmark paper 来说，人工构建和人工检查比纯 LLM 合成更容易获得信任。

4. **rebuttal 有效补强**

   作者在 rebuttal 中回应了数据规模、few-shot、错误分析、可复现性等问题，使得原本部分 reviewer 的担忧下降。

#### Rebuttal 结果

这篇的 rebuttal 属于“有效补证据”的类型。作者针对数据规模、few-shot 设置、错误分析、benchmark release / reproducibility 等问题做了补充，使得审稿人更相信它不仅是一个小型测试集，而是有清晰任务定义和分析价值的 benchmark。

最终结果是接收为 poster。它没有进一步冲到更高层级，主要还是因为规模、覆盖范围和方法新意有限；但 rebuttal 已经足够让 AC 认为核心贡献成立。

#### 仍然存在的问题

- 数据规模不大。
- 覆盖范围仍有限。
- 方法论贡献偏 benchmark / evaluation，本身不算特别 radical。

#### 对我们的启发

如果做文档 benchmark，最好不要只给答案。可以设计：

- supporting evidence span / bbox
- reasoning path
- intermediate OCR evidence
- page-level grounding
- answer provenance

这样 benchmark 才能解释模型为什么错。

---

### 3.2 DAVE

论文：DAVE: A VLM Vision Encoder for Document Understanding and Web Agents  
OpenReview：https://openreview.net/forum?id=kgk0NqjsoW  
结果：Accept Poster  
原始评分：8 / 4 / 4 / 4  
AC 判断 rebuttal 后约提升到：8 / 6 / 6- / 6-

#### 核心贡献

DAVE 针对文档理解和 web agent 任务训练一个专用 vision encoder。它结合：

- self-supervised pretraining on unlabeled images
- supervised autoregressive pretraining
- model merging
- ensemble training

评测覆盖 document understanding、VQA、web localization、agent benchmark 等多个任务。

#### 为什么能中

1. **实际需求明确**

   现有通用 vision encoder 不一定适合文档、网页、UI、OCR-heavy 场景。这个切入点很现实。

2. **结果覆盖面广**

   它不是只在一个数据集上涨点，而是在多个 document / web agent 任务上做系统验证。

3. **rebuttal 补上关键实验**

   主要质疑包括：

   - 跨领域泛化是否成立
   - 与其他 visual encoder 对比是否充分
   - 是否依赖 SigLIP2
   - 方差与稳定性如何

   作者通过补实验缓解了这些担忧。

#### Rebuttal 结果

DAVE 的 rebuttal 明显改变了审稿走势。初始评分里有多个 4 分，主要担心 novelty、跨域泛化、与现有 visual encoder 的区别、是否依赖 SigLIP2、结果方差等。

作者补充实验和解释后，AC 判断低分 reviewer 大概率会上调到 6 左右，整体从“有强结果但证据不足”变成“虽然是组合式方法，但实证价值足够”。最终接收为 poster。剩下没有完全消掉的是 novelty：它仍被看作若干已有训练技巧的有效组合，而不是全新的模型范式。

#### 仍然存在的问题

- 技术新意被认为是若干已有方法的组合。
- 一些 general VQA 指标可能有回退。
- 仍有 reviewer 认为 novelty 不够强。

#### 对我们的启发

文档方向的模型文章如果技术不够“新”，就必须用强实验体系补足：

- 多任务覆盖
- 多 backbone
- 多数据域
- ablation
- variance
- efficiency
- failure analysis

否则很容易被评价为“工程组合”。

---

### 3.3 TableMaster

论文：TableMaster: A Recipe to Advance Table Understanding with Language Models  
OpenReview：https://openreview.net/forum?id=YyPZPrPjQD  
结果：Accept Poster  
评分：6 / 4 / 4，AC 表示 rebuttal 后 reviewer 分数大概率上调

#### 核心贡献

TableMaster 针对表格理解提出一个组合式 recipe，拆解出四类核心挑战：

- data localization
- weak table semantics
- numerical errors
- rigid symbolic reasoning

并对应设计：

- table-of-focus
- table verbalization
- textual / symbolic adaptive reasoning

#### 为什么能中

1. **问题拆解非常清楚**

   它不是泛泛说“表格理解难”，而是把难点拆成 reviewer 可以理解和验证的几个模块。

2. **recipe 是 cohesive 的**

   虽然每个组件不一定全新，但组合后形成了清晰 pipeline。

3. **实验覆盖多个 benchmark 和模型**

   审稿人喜欢这种“不是只对一个模型有效”的验证方式。

4. **rebuttal 补了缺口**

   初审中 reviewer 质疑 ablation、latency、novelty、model scale。作者在 rebuttal 中补了实验和解释。

#### Rebuttal 结果

TableMaster 的 rebuttal 结果是“从边缘变成可接收”。初始分数并不高，多个 reviewer 认为组件本身比较常见，缺少足够 ablation、latency 和 model-scale 分析。

作者在 rebuttal 中补充了实验和解释后，AC 明确表示 reviewer 分数大概率会整体上调。最后接受的关键不是它每个模块都新，而是 reviewer 接受了它作为一个 cohesive recipe 的价值：问题拆解清楚，实验覆盖比较完整，对表格理解实践有参考意义。

#### 仍然存在的问题

- 单个技术组件的新意有限。
- 更像 recipe / system paper，而不是理论或算法突破。
- 初始版本的 ablation 和 latency 分析不够。

#### 对我们的启发

如果 benchmark 面向表格 / 文档解析，可以借鉴这种“问题分解式 benchmark”：

- localization challenge
- structure challenge
- semantic challenge
- numeric reasoning challenge
- cross-page evidence challenge
- multimodal grounding challenge

比单纯按任务名分类更有诊断价值。

---

### 3.4 ChartGalaxy

论文：ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation  
OpenReview：https://openreview.net/forum?id=P4lFbvZ4HH  
结果：Accept Poster，conditional acceptance 后满足条件  
评分：6 / 6 / 8 / 6

#### 核心贡献

ChartGalaxy 是一个大规模 infographic chart 数据集，包括：

- 1,701,356 个 programmatic charts
- 61,833 个 real infographic charts
- 75 种 chart types
- 440 种 variations
- 68 个 layout templates

它支持 chart understanding、code generation、example-based generation 等多个用途。

#### 为什么能中

1. **规模和多样性明显**

   这类 dataset paper 的基础分来自数据本身的稀缺性、规模和覆盖面。

2. **用途不止一个**

   数据集可以服务理解、生成、代码生成、微调、benchmark 等任务，价值面更宽。

3. **rebuttal 回应了 QA 与 metric 问题**

   reviewer 关注 template details、QA quality、metric reliability、URL correctness 等问题。作者补充解释后被接受。

4. **伦理和版权问题被认真处理**

   这篇有 conditional acceptance，要求作者补充版权、terms、legal opinion 等证据。

#### Rebuttal 结果

ChartGalaxy 的 rebuttal 主要解决了数据集论文常见的可信度问题：template 设计、QA 质量、评测 metric 是否可靠、链接和数据来源是否正确等。reviewer 对数据规模和多样性原本评价较正面，rebuttal 起到的是“补齐可信度缺口”的作用。

最终是 conditional acceptance。条件集中在版权、数据使用条款和法律风险上。也就是说，技术和数据贡献基本被认可，但 release 合规性必须进一步证明。这个 case 对 benchmark / dataset paper 很有参考价值：伦理、版权、发布权限会直接进入最终决策。

#### 仍然存在的问题

- 真实图表来源涉及版权和 terms 风险。
- 自动评测指标可靠性仍需人类评估支撑。
- 合成模板是否覆盖真实分布仍是潜在问题。

#### 对我们的启发

如果 benchmark 使用网页、PDF、图表、文档截图，需要提前准备：

- 数据来源许可说明
- robots / terms 检查
- redistribution policy
- takedown policy
- license
- anonymization / privacy risk
- 是否只发布 derived metadata

这类问题现在已经不是附属项，而是可能影响接收结果。

---

### 3.5 Visual Self-Refine

论文：Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing  
OpenReview：https://openreview.net/forum?id=RI0oNr1b0y  
结果：Accept Poster  
评分：6 / 6 / 4 / 6，rebuttal 后 reviewer 倾向上调

#### 核心贡献

Visual Self-Refine 让模型先生成 pixel-level localization outputs，再把这些输出可视化并反馈给模型自我修正。作者将其用于 chart parsing，并构造 ChartP-Bench。

#### 为什么能中

1. **机制有可解释的操作性**

   它不是简单 prompt，而是让模型显式产生可视化中间结果，再利用视觉反馈修正。

2. **rebuttal 补了强基线**

   初审质疑包括：

   - 缺少 Qwen3 / InternVL / Gemini 2.5 等新模型对比
   - 方法是否只是短期 workaround
   - 是否会被下一代 VLM 自然解决

   作者补充新模型对比后，结果仍有优势。

3. **benchmark 与方法互相支撑**

   ChartP-Bench 为方法要解决的问题提供了更明确的评测场景。

#### Rebuttal 结果

Visual Self-Refine 的 rebuttal 重点补了强基线。初审时，reviewer 担心文章没有和 Qwen3、InternVL、Gemini 2.5 等更新的视觉推理模型比较，也担心这个方法只是当前模型能力不足时的临时 workaround。

作者补充新模型对比后，证明 pixel-guided self-refinement 仍然有增益。这个 rebuttal 比较成功，因为它正面回应了“是不是已经被新模型解决了”这一核心威胁。最终 AC 认可其机制和实验补充，接收为 poster。

#### 仍然存在的问题

- 方法比较 task-specific。
- 额外 self-refine 调用增加成本。
- 在部分旧数据集上提升不稳定。

#### 对我们的启发

文档解析 benchmark 如果能配套设计“中间证据格式”，会更强：

- bbox
- cell coordinates
- reading order
- relation graph
- visual trace
- evidence mask
- rendered intermediate output

这能让 benchmark 兼具评测与调试价值。

---

## 4. 拒稿论文复盘

### 4.1 HalluText

论文：HalluText: Towards Benchmarking and Mitigating OCR Hallucination for LVLMs  
OpenReview：https://openreview.net/forum?id=LRnt6foJ3q  
结果：Reject  
评分：2 / 4 / 4 / 4

#### 核心想法

HalluText 试图评估和缓解 LVLM 的 OCR hallucination 问题。方向本身很重要，因为 text-rich image 中模型常常会读错、编造或混淆文字。

#### 主要拒稿原因

1. **benchmark 构建细节不够扎实**

   Reviewer 4VgK 和 ne8t 都指出，作为 benchmark paper，数据构建过程交代得不够：

   - 如何清洗数据？
   - QA 如何生成？
   - distractor 怎么构造？
   - distractor 是否公平？
   - Position 类别的歧义如何处理？
   - 是否有多标注者一致性？
   - 合成数据是否贴近真实 OCR 错误？
   - 如何过滤低质量样本？

   这些问题对 benchmark paper 很致命，因为 benchmark 的核心资产就是“可信的问题定义和可信的数据”。

2. **对外部 OCR 的依赖没有充分处理**

   方法依赖外部 OCR，但没有充分分析外部 OCR 错误如何影响最终结果。

3. **对照实验不够强**

   reviewer 期待看到更多：

   - calibration baseline
   - consensus decoding
   - contrastive decoding variants
   - MinerU / MinerU2.5
   - OmniDocBench
   - 更强 document parser pipeline

4. **泛化范围偏窄**

   任务更多集中在 OCR / recognition / multiple-choice 层面，较少覆盖 open-ended DocQA、ChartQA、multi-step reasoning 等场景。

5. **表述有 overclaim 风险**

   审稿人觉得有些结论讲得比实验支撑更强。

#### Rebuttal 结果

HalluText 的 rebuttal 只部分缓解了问题。作者确实回应了一些 reviewer 质疑，也有 reviewer 表示部分顾虑被解决、存在上调意愿。但最终 AC 仍然认为核心问题没有被充分消除。

没有被完全解决的点主要包括：benchmark 构建细节仍不够透明，外部 OCR 依赖和错误传播分析不足，KL-divergence 等方法设计的必要性解释不够，计算成本和替代 baseline 比较也不够扎实。最终结果仍是 reject。

这个 case 的关键信号是：benchmark 论文的 rebuttal 不能只补文字解释。只要数据构建、标注质量、distractor 公平性、合成数据真实性这些根问题没有用数字和流程补硬，方向再重要也很难翻盘。

#### 复盘判断

HalluText 不是输在问题不重要，而是输在 benchmark 的信任链不完整。它提出了一个有价值的问题，但没有让 reviewer 充分相信：

- 数据是怎么来的；
- 样本为什么公平；
- 错误类型为什么代表真实场景；
- 标注为什么可靠；
- 评测结果为什么能推广。

---

### 4.2 Chain-of-Reading

论文：End-to-End Document Understanding via Chain-of-Reading  
OpenReview：https://openreview.net/forum?id=6YXMyPrDEN  
结果：Reject  
评分：6 / 6 / 4 / 4

#### 核心想法

文章提出通过 Chain-of-Reading 做端到端文档理解，让模型在文档中定位、读取并推理。

#### 主要拒稿原因

1. **泛化性不足**

   审稿人担心实验主要集中在 academic / government reports，无法证明方法适用于更多真实文档类型。

2. **novelty 被质疑**

   locate-then-reason 的流程与 DocReact、CoT、agentic workflow 等已有思路相似。

3. **缺少关键 baseline**

   reviewer 希望看到：

   - few-shot prompting baseline
   - MinerU 等现代 parser
   - 更强 pipeline baseline
   - 非 Qwen backbone
   - general VQA / broader document tasks

4. **成本分析不足**

   Chain-of-Reading 生成长 reasoning trace，会带来 token 和 latency 成本。作者承诺补充，但当前版本没有给出足够证据。

#### Rebuttal 结果

Chain-of-Reading 的 rebuttal 有解释效果，但没有形成足够强的证据反转。作者对“是否真的 end-to-end”“是否只是 pipeline / DocReact / CoT 类方法变体”等问题做了说明，也解释了模型如何利用 raw pixels 和内部化读取过程。

但 AC 更看重当前提交中已经存在的证据。关键实验仍停留在承诺层面，例如更完整的 token / latency 成本、现代 parser baseline、few-shot baseline、更多 backbone 和更广泛文档类型。最终 AC 认为这些缺口不能留到 final version，因此 reject。

#### 复盘判断

这篇的问题在于：它要证明“端到端读文档优于 pipeline”，但证据链没有闭合。尤其当方法引入更长推理过程时，必须证明性能收益值得额外成本。

---

### 4.3 Structured Attention Matters

论文：Structured Attention Matters to Multimodal LLMs in Document Understanding  
OpenReview：https://openreview.net/forum?id=3OnJAvuxd3  
结果：Reject  
评分：4 / 2 / 2 / 2

#### 核心想法

文章认为结构化文本表示能帮助 MLLM 做文档理解，比如把文档内容转成 LaTeX-like 或结构化格式。

#### 主要拒稿原因

1. **核心观察不够新**

   “纯文本会丢失结构信息”被认为是已有共识。

2. **方法像 prompt / formatting trick**

   使用 LaTeX-like formatting 被 reviewer 看作简单工程技巧，而不是强技术贡献。

3. **结构生成器细节不足**

   审稿人不清楚 structured text generator 如何构建，质量如何控制。

4. **缺少强 baseline**

   reviewer 期待 HTML / XML 等结构化格式对比。

5. **attention 分析证据不足**

   attention visualization 不能直接证明因果机制，需要更严谨的 causal / intervention 分析。

#### Rebuttal 结果

这篇没有看到有效的 author rebuttal 来系统回应 reviewer 的核心担忧，因此初审意见基本原样保留。低分 reviewer 对 novelty、方法定义、baseline 和 attention 解释的质疑没有被补实验或澄清扭转。

最终 reject 的原因很直接：如果一篇文章被认为只是“结构化格式输入”或 prompt formatting trick，那么 rebuttal 至少需要补 HTML / XML / Markdown / LaTeX 等格式对照、token / latency 成本、结构生成器质量，以及 causal evidence。这里这些关键补救没有发生。

#### 复盘判断

这篇的教训是：如果方法看起来像“换一种格式喂给模型”，就必须做非常强的机制验证和 baseline 对比。否则容易被归为 prompt engineering。

---

### 4.4 GDI-Bench

论文：GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling  
OpenReview：https://openreview.net/forum?id=4l8QRqYzH9  
结果：Reject  
评分：6 / 2 / 6 / 6

#### 核心想法

GDI-Bench 试图区分 document intelligence 中的视觉复杂度和推理复杂度，并提出 benchmark 和相关 fine-tuning 方法。

#### 主要拒稿原因

1. **标注质量审计不足**

   reviewer 关心：

   - annotation quality
   - inter-annotator agreement
   - remaining bias
   - noise rate
   - human verification

2. **复杂度 taxonomy 不够稳**

   视觉复杂度被质疑是由模型 performance gap 推出来的，而不是由数据本身的 intrinsic property 定义。

3. **benchmark 与方法有些脱节**

   文章既做 benchmark，又提出 fine-tuning 方法，但二者之间的动机和必要性没有完全连起来。

4. **覆盖范围有限**

   主要是 single-image document understanding，泛化到多页、多文档、真实业务场景的证据不足。

#### Rebuttal 结果

GDI-Bench 的 rebuttal 有一定澄清作用。作者尝试解释视觉复杂度与推理复杂度的定义，也补充说明 R0 / R1 / R2 等 reasoning complexity 分级，以及 benchmark 与 fine-tuning 方法之间的关系。

但这些解释没有完全解决 AC 的核心担忧：标注质量审计不足，inter-annotator agreement 和 bias / noise 证据不够；视觉复杂度 taxonomy 仍被怀疑不是充分基于数据内在属性；benchmark 和方法贡献之间仍显得有些分离；任务覆盖也偏 single-image。最终虽然有多个 6 分，仍被 reject。

#### 复盘判断

这篇说明：即使 reviewer 给了多个 6 分，benchmark paper 仍可能因为“定义不够可靠”被拒。尤其是 taxonomy / difficulty level，必须基于可解释、可复核的样本属性，而不能只依赖模型表现反推。

---

### 4.5 MDocAgent

论文：MDocAgent: A Multi-Modal Multi-Agent Framework for Document Question Answering  
OpenReview：https://openreview.net/forum?id=05SHW9ai9e  
结果：Reject  
评分：4 / 4 / 2 / 4

#### 核心想法

文章提出多模态多智能体框架，用于文档问答，整合文本与视觉线索。

#### 主要拒稿原因

1. **novelty 弱**

   reviewer 认为系统主要是已有 retriever、prompting、RAG、多 agent 组件的组合。

2. **过度依赖 GPT-4o**

   缺少更广泛的模型验证，无法证明框架本身的普适性。

3. **错误分析不足**

   没有系统说明失败案例、错误来源、不同 agent 的贡献。

4. **效率分析缺失**

   多 agent 通常带来更高时间、token、memory 成本，但文章没有充分量化。

#### Rebuttal 结果

MDocAgent 没有看到有效 rebuttal 来回应 reviewer 的主要质疑。由于 novelty、GPT-4o 依赖、error analysis 和 efficiency 都是系统类 / agent 类论文的核心问题，缺少 rebuttal 基本意味着低分意见不会被扭转。

最终 reject 的逻辑是：多 agent 框架本身不稀缺，必须证明每个 agent 的必要性、相对 strong RAG / parser baseline 的收益，以及额外成本的合理性。这些在 rebuttal 阶段没有补起来。

#### 复盘判断

文档 QA 方向的 agent paper 很容易被认为是拼装系统。要避免这个问题，必须证明：

- 每个 agent 有不可替代贡献；
- 比单 agent / strong RAG / strong parser 明显更好；
- 额外成本值得；
- failure modes 被系统分析过。

---

### 4.6 CoTabBench

论文：CoTabBench: A Real-World Benchmark for Question Answering over Weakly-Structured and Heterogeneous Tables  
OpenReview：https://openreview.net/forum?id=wcInjlUp8V  
结果：Reject  
评分：2 / 4 / 4 / 6

#### 核心想法

CoTabBench 面向真实弱结构、异构表格问答，评估 closed / open / thinking / non-thinking models。

#### 主要拒稿原因

1. **数据构建细节不足**

   reviewer 认为 eval set 和 training set 的构建过程都不够清楚。

2. **train / eval overlap 风险**

   CoTabBench 和 CoTabInstruct 之间可能存在重叠，导致性能被高估。

3. **采样流程不清楚**

   arXiv / web 表格如何采样、筛选、去重，没有充分说明。

4. **LLM 合成 QA 缺少人工验证**

   如果 QA pair 主要由 LLM 生成，就需要人类验证、错误率估计、一致性分析。

5. **没有 author response**

   关键担忧没有在 rebuttal 中被修复。

#### Rebuttal 结果

CoTabBench 没有 author response，因此 reviewer 关于数据构建、train / eval overlap、采样流程、LLM 合成 QA 人工验证不足等担忧全部保留。

这对 benchmark paper 尤其伤，因为 reviewer 质疑的是“数据是否可信”这类根问题。没有 rebuttal 的情况下，即便有一个 reviewer 给到 6 分，也很难让 AC 忽略数据信任链上的缺口。

#### 复盘判断

真实弱结构表格是很好的方向，但 benchmark 文章不能只强调“real-world”。越是真实、混乱、异构的数据，越需要更严格的数据审计和泄漏控制。

---

### 4.7 ChartNexus

论文：ChartNexus: Evaluating Multi-Chart Reasoning Capabilities of Multimodal Large Language Models  
OpenReview：https://openreview.net/forum?id=xg0fmtqh8d  
结果：Reject  
评分：2 / 6 / 4 / 2

#### 核心想法

ChartNexus 评估 MLLM 的 multi-chart reasoning 能力。

#### 主要拒稿原因

1. **novelty 和 insight 不足**

   reviewer 认为文章更多是在说明“多图表更难”，但这本身不够新。

2. **LLM-as-judge 质量没有充分讨论**

   开放式问答如何评估、judge 是否可靠、是否有人类校验，都需要更强证据。

3. **开放式 QA 分歧处理不清楚**

   多个答案、部分正确、推理路径不同的情况如何处理，没有充分说明。

4. **数据分布可能偏**

   比如 bar chart 可能过多，影响 benchmark 代表性。

5. **缺少细粒度分析**

   reviewer 希望看到按 chart type、reasoning type、visual complexity、question type 的结果拆解。

6. **data leakage mitigation 不够有说服力**

   benchmark 是否被模型见过、如何排查泄漏，是关键问题。

#### Rebuttal 结果

ChartNexus 的 rebuttal 回应了一部分 reviewer 问题，包括数据规模理解、multi-chart 任务动机、LLM-as-judge、data leakage 和部分数据分布问题。但 AC 的判断是：这些回应没有把论文从“一个更难的 benchmark”提升到“有足够新 insight 的 benchmark”。

最终 reject 的关键不是多图表方向不重要，而是 reviewer 仍没有看到足够清晰的新发现。比如 context-vision gap、分辨率瓶颈、模型在多图时性能下降等，被认为大体符合预期；3D chart 等观察有趣，但不足以支撑整篇 benchmark 的贡献。

#### 复盘判断

benchmark 不能只证明任务困难。它还要回答：

- 困难来自哪里？
- 哪些模型在哪些能力上失败？
- 新 benchmark 相比已有 benchmark 多了什么诊断价值？
- 对未来方法有什么指导？

---

### 4.8 VisR-Bench

论文：VisR-Bench: An Empirical Study on Visual Retrieval-Augmented Generation for Multilingual Long Document Understanding  
OpenReview：https://openreview.net/forum?id=7iFZ6uzILL  
结果：Reject  
评分：4 / 2 / 6 / 6

#### 核心想法

VisR-Bench 关注 multilingual long document understanding 中的 visual retrieval-augmented generation。方向结合了长文档、多语种、视觉检索和 RAG，问题很现实。

#### 主要拒稿原因

1. **强模型覆盖不足**

   初始版本缺少 OpenAI o3、Gemini 2.5 等 reasoning-capable MLLMs。作者在 rebuttal 中补了，但没有完全扭转 reviewer 对贡献强度的判断。

2. **过度强调 Top-1 retrieval**

   reviewer 认为只看 Top-1 retrieval metric 不足以说明 end-to-end long document QA 能力。

3. **合成 QA 缺少足够人工验证**

   作者补充了 50 个 QA pair 的人工检查，但 reviewer 仍担心规模太小，无法支撑整个 benchmark 的可靠性。

4. **数据构建偏差**

   LLM-based filtering / heuristics 可能引入偏差，但文章没有充分 error analysis。

5. **语言覆盖不足**

   排除了中文等 logographic language，使 multilingual 的说服力下降。

#### Rebuttal 结果

VisR-Bench 的 rebuttal 补了不少内容，包括加入 OpenAI o3、Gemini 2.5 等强 reasoning-capable MLLM 结果，补充 50 个 QA pair 的人工检查，给出更多 multilingual 数据统计，并解释中文等语言缺失与语料限制有关。

这些补充让部分 reviewer 认可文章的实证价值，但没有完全改变最终判断。AC 仍认为合成 QA 的人工验证规模偏小，LLM-based heuristic 带来的偏差缺少系统 error analysis，Top-1 retrieval 不能充分代表 end-to-end RAG 能力，多语言覆盖与公平性不足也削弱了 benchmark 的定位。最终 reject。

#### 复盘判断

这篇的方向有价值，但 benchmark 贡献被认为不够 decisive。对于 multilingual benchmark，“语言覆盖”和“跨语言公平性”本身就是核心贡献的一部分，不能作为边缘限制轻轻带过。

---

### 4.9 DocPruner

论文：DocPruner: A Storage-Efficient Framework for Multi-Vector Visual Document Retrieval via Adaptive Patch-Level Embedding Pruning  
OpenReview：https://openreview.net/forum?id=mEMGL1fLOO  
结果：Reject  
评分：6 / 6 / 2 / 2

#### 核心想法

DocPruner 试图降低 visual document retrieval 中 multi-vector embedding 的存储开销，通过 adaptive patch-level embedding pruning 保留重要 patch。

#### 主要拒稿原因

1. **问题重要，但方法创新有限**

   attention-based pruning 被认为是比较标准的思路。

2. **没有证明 adaptive threshold 优于简单 fixed-ratio baseline**

   这是最关键的实验缺口。如果一个复杂方法不能明显超过简单基线，就很难成立。

3. **实验图表和数据点不够清晰**

   reviewer 质疑 baseline plots、missing data points、interpolation 等问题。

4. **rebuttal 未解决核心质疑**

   作者没有充分证明关键设计的必要性。

#### Rebuttal 结果

DocPruner 的 rebuttal 没有解决最关键的实验有效性质疑。作者尝试解释 adaptive pruning 和实验设置，但 reviewer 仍认为没有证明它明显优于更简单的 fixed-ratio attention pruning baseline。

因此最终 reject 的核心不是“压缩 visual document retrieval embedding”这个问题不重要，而是 trade-off 证据不够硬。对于这类 efficiency paper，rebuttal 如果不能补出完整曲线和强 baseline 对比，很难说服 reviewer 接受更复杂的机制。

#### 复盘判断

这篇不是方向错，而是实验有效性没有打穿。对于 compression / pruning / retrieval efficiency 这类论文，最重要的是：

- 和最简单的强 baseline 比；
- 画出完整 trade-off curve；
- 不只报一个点；
- 证明复杂机制带来的收益超过复杂度。

---

## 5. 中稿与拒稿的核心差异

### 5.1 中稿论文通常做对了什么

#### 第一，问题定义有清楚边界

中稿论文往往能一句话说清楚自己评估或解决的能力：

- OCR-Reasoning：复杂 text-rich image reasoning
- DAVE：文档和 web agent 场景的专用 vision encoder
- TableMaster：表格理解中的定位、语义、数值、符号推理
- ChartGalaxy：infographic chart understanding and generation
- Visual Self-Refine：chart parsing 中的 pixel-guided self-refinement

它们不是泛泛说“文档理解很难”，而是把问题切成 reviewer 可以判断的具体能力。

#### 第二，数据或方法有可诊断结构

好的 benchmark 不只是一个测试集，而是带结构的诊断工具：

- 按能力分类
- 按错误类型分类
- 有 intermediate evidence
- 有 reasoning trace
- 有 bbox / cell / layout / chart element grounding
- 有真实和合成的分层分析

#### 第三，rebuttal 提供了真实新证据

中稿论文的 rebuttal 往往补了：

- 新 baselines
- 新 ablations
- human validation
- metric reliability analysis
- latency / token cost
- error analysis
- ethics / copyright clarification

而不是只说“我们会在 final version 补”。

#### 第四，实验覆盖面足够宽

中稿论文通常覆盖多个维度：

- 多模型
- 多数据集
- 多任务
- 多语言或多 domain
- open-source + proprietary
- old baseline + latest strong model

---

### 5.2 拒稿论文通常卡在哪里

#### 第一，benchmark trust chain 断裂

常见问题包括：

- 数据来源不清楚
- 清洗规则不清楚
- LLM 合成 QA 没有人工验证
- distractor 质量不清楚
- 标注一致性缺失
- taxonomy 主观
- difficulty level 由模型表现反推
- train / eval overlap 没控制
- data leakage 排查不足

这些问题会让 reviewer 觉得 benchmark 不可依赖。

#### 第二，novelty 被归为 obvious composition

很多方法会被评价为：

- prompt engineering
- formatting trick
- RAG pipeline
- agent orchestration
- attention pruning
- parser + LLM 拼装

这些不是不能发，但必须证明组合后产生了非显然收益，并且强过简单 baseline。

#### 第三，baseline 跟不上审稿人预期

文档方向 reviewer 越来越期待看到：

- 最新 MLLM
- proprietary strong model
- open-source strong model
- modern parser
- OCR pipeline
- strong RAG
- strong prompt baseline
- ablation against obvious variants

如果 baseline 缺失，哪怕方向很好，也容易被认为证据不足。

#### 第四，成本没有算清楚

文档解析任务常常涉及：

- 多页输入
- 高分辨率图像
- OCR
- layout parsing
- retrieval
- multi-agent
- multi-step reasoning

因此 reviewer 会自然关心：

- latency
- token cost
- GPU memory
- storage
- index size
- throughput
- API cost

如果方法提升不大但成本明显更高，就很危险。

#### 第五，benchmark 没有产生新 insight

“模型在我们的 benchmark 上表现不好”不够。benchmark paper 更应该回答：

- 哪类视觉结构最难？
- 哪类推理最难？
- OCR 错误如何传播？
- layout 错误如何影响 QA？
- retrieval 错误和 generation 错误如何区分？
- 哪些模型失败模式不同？
- 哪些能力是现有 benchmark 没测到的？

---

## 6. 对你做 benchmark 的具体建议

### 6.1 先写清楚 benchmark 要定义什么问题

不要从“我要收一个数据集”开始，而要从“我要定义一个可靠评测问题”开始。

建议用下面这个模板：

```text
现有 benchmark 主要评估 ________，
但真实文档场景中还存在 ________。
这个能力无法被现有 benchmark 可靠衡量，因为 ________。
因此我们提出 ________，它系统评估 ________。
```

一个好的 benchmark 问题最好满足：

- 真实存在
- 现有 benchmark 测不到或测不准
- 可以被清晰分解
- 可以设计客观评测协议
- 能产生诊断性结论

---

### 6.2 数据构建要写到可以复现

benchmark paper 的数据部分应该写得像实验协议，而不是故事描述。

至少需要包括：

- 原始数据来源
- 采样策略
- license / terms
- 去重规则
- 清洗规则
- OCR / parser 预处理工具版本
- QA 生成流程
- distractor 生成流程
- 人工标注界面或说明
- 标注者数量
- 标注者培训
- disagreement 解决方式
- inter-annotator agreement
- 样本过滤规则
- 质量抽检比例
- 错误率估计
- train / dev / test 划分
- leakage 检查

如果其中一些不能公开，也要解释为什么，以及公开哪些 derived artifacts。

---

### 6.3 合成数据要证明贴近真实错误

如果 benchmark 使用合成数据，需要回答：

- 合成规则来自哪里？
- 是否统计过真实 OCR / parser 错误分布？
- synthetic error 和 real error 的分布是否接近？
- 是否有 real-only subset？
- 模型在 synthetic 和 real subset 上表现是否一致？
- 是否有人类判断合成样本自然性？

可以考虑设计：

- Real subset
- Synthetic controlled subset
- Hard negative subset
- Long-tail subset
- Human-written subset

这样既能控制变量，又能证明真实相关性。

---

### 6.4 QA 和 distractor 是 benchmark 成败关键

很多 benchmark 被拒不是因为题目少，而是因为 QA 不可信。

需要重点说明：

#### 对 QA

- question 是否唯一可答？
- answer 是否可从文档直接验证？
- 是否需要外部知识？
- 是否存在多答案？
- 是否存在 ambiguous wording？
- 是否记录 evidence？
- 是否按 reasoning type 分类？

#### 对 distractor

- distractor 是否过于明显？
- 是否只靠 language prior 就能排除？
- 是否长度、格式、语义相近？
- 是否来自同页 / 同文档相似区域？
- 是否会引入多个正确答案？
- 是否经过人工检查？

如果是 multiple-choice benchmark，distractor 公平性尤其重要。

---

### 6.5 taxonomy 不能只靠模型表现定义

难度分类最好基于样本内在属性，而不是“模型错得多所以难”。

例如可以按这些维度定义：

- OCR density
- layout complexity
- number of pages
- evidence span length
- cross-page dependency
- table nesting
- chart type
- visual clutter
- answer type
- reasoning hop count
- arithmetic operations
- entity linking requirement
- spatial relation requirement

然后再分析模型在不同维度上的 performance gap。

这比先看模型哪里掉点再定义类别更可靠。

---

### 6.6 baseline 要覆盖“最容易被 reviewer 想到”的方案

建议至少包括：

- OCR + LLM
- parser + LLM
- layout-aware parser + LLM
- RAG + LLM
- direct MLLM
- high-resolution MLLM
- latest open-source MLLM
- latest proprietary MLLM
- simple prompt baseline
- few-shot baseline
- task-specific method
- human upper bound or human subset

文档解析方向尤其要注意现代 parser：

- MinerU / MinerU2.5
- OmniDocBench 相关 parser / baseline
- Nougat / Marker / PaddleOCR / Docling 等可按任务选择

不用所有都上，但必须解释为什么选择这些 baseline。

---

### 6.7 评测指标要能区分错误来源

只报 accuracy / F1 不够。建议把错误拆开：

- OCR error
- localization error
- retrieval error
- layout parsing error
- reasoning error
- arithmetic error
- hallucination
- answer formatting error
- judge error

如果是开放式 QA，还要处理：

- exact match 过严
- semantic match 不稳定
- LLM-as-judge 偏差
- partial credit
- 多答案问题

可以用：

- rule-based metric
- human evaluation subset
- LLM judge calibration
- judge agreement
- adversarial judge audit
- evidence correctness score

---

### 6.8 必须准备一组“新 insight”图表

benchmark 论文最好提前设计这些分析：

- performance by task type
- performance by document type
- performance by page count
- performance by OCR density
- performance by layout complexity
- performance by reasoning type
- performance by answer type
- retrieval recall vs final QA accuracy
- parser quality vs final QA accuracy
- hallucination rate by condition
- model family comparison
- error transition diagram

这些图表能让文章从“我们做了一个数据集”变成“我们解释了一个问题”。

---

## 7. Benchmark 文章写作建议

### 7.1 Introduction 要避免空泛

不要只写：

```text
Document understanding is important. Existing MLLMs still struggle. We propose a benchmark.
```

更好的结构是：

```text
真实场景中的某个具体能力很重要。
现有 benchmark 无法隔离或可靠评估这个能力。
我们定义了一个可验证的任务设置。
我们通过数据构建、人工标注和诊断分析证明该任务有价值。
结果显示当前模型在某些具体能力上系统性失败。
```

---

### 7.2 Dataset section 要写得很硬

建议小节结构：

1. Source Documents
2. Sampling Strategy
3. Preprocessing and Cleaning
4. Task Taxonomy
5. QA / Annotation Pipeline
6. Human Verification
7. Quality Control
8. Train / Dev / Test Split
9. Leakage Control
10. Ethics and License

每个小节最好有数字，不要只有描述。

---

### 7.3 Experiments section 要回答 reviewer 的默认问题

默认问题包括：

- 为什么这些 baseline 足够强？
- 为什么这个 benchmark 不是被语言先验解决？
- 为什么不是 OCR 错误导致所有结果？
- 为什么不是 prompt 没调好？
- 为什么不是 parser 换一个就解决？
- 为什么不是模型看过数据？
- 为什么这个 benchmark 对未来研究有指导意义？

---

### 7.4 Rebuttal 前就要准备好补实验

从这些 case 看，rebuttal 的作用很大。建议投稿前就准备一个“rebuttal experiment bank”：

- 1-2 个最新模型结果
- 1 个强 parser baseline
- 1 个 efficiency table
- 1 个 human validation table
- 1 个 leakage check
- 1 个 ablation
- 1 个 error analysis
- 1 个 qualitative case

这样 reviewer 一问，就能立即补证据。

---

## 8. 可以直接套用的 Benchmark Checklist

### 8.1 数据可信度

- [ ] 原始数据来源清楚
- [ ] license / terms 清楚
- [ ] 采样策略清楚
- [ ] 去重方法清楚
- [ ] 清洗规则清楚
- [ ] train / test leakage 检查完成
- [ ] LLM 合成部分有人工验证
- [ ] 标注者数量与一致性有报告
- [ ] 错误率有估计
- [ ] 数据分布有统计

### 8.2 任务定义

- [ ] 和现有 benchmark 的差异明确
- [ ] 每个 task type 有定义
- [ ] 难度分类基于内在属性
- [ ] 每个问题有可验证 evidence
- [ ] ambiguous cases 有处理规则
- [ ] open-ended answer 有评测协议
- [ ] multiple-choice distractor 有公平性检查

### 8.3 实验设计

- [ ] direct MLLM baseline
- [ ] OCR / parser + LLM baseline
- [ ] RAG baseline
- [ ] latest open-source model
- [ ] latest proprietary model
- [ ] few-shot / prompt baseline
- [ ] ablation
- [ ] human or expert subset
- [ ] efficiency cost
- [ ] error analysis

### 8.4 文章贡献

- [ ] 不只是数据集，还有诊断 insight
- [ ] 不只是掉点，还有错误来源分析
- [ ] 不只是 leaderboard，还有 taxonomy
- [ ] 不只是 synthetic，还有 real-world validation
- [ ] 不只是 current models，还有未来研究方向

---

## 9. 最值得借鉴的写法

### 9.1 从 OCR-Reasoning 学

把 final answer 和 reasoning trace 分开标注。这样 benchmark 能分析：

- 读错了
- 找错证据了
- 推理错了
- 答案格式错了

这比只报 accuracy 强很多。

### 9.2 从 TableMaster 学

先拆问题，再做方法或 benchmark。比如文档解析可以拆成：

- visual acquisition
- text recognition
- layout reconstruction
- element linking
- evidence retrieval
- reasoning
- answer generation

每一环都可以有对应评测。

### 9.3 从 ChartGalaxy 学

如果做大规模数据，必须同时处理：

- scale
- diversity
- quality
- copyright
- metric reliability

规模大但质量和授权说不清，会很危险。

### 9.4 从 HalluText 学

benchmark 文章最怕“数据怎么来的”讲不清。哪怕问题很重要，只要构建流程不透明，reviewer 就很难信任结论。

### 9.5 从 VisR-Bench 学

多语言 benchmark 不能只是把英语数据翻译或扩展到几种语言。语言覆盖本身就是贡献，需要考虑：

- writing system
- document layout convention
- OCR difficulty
- corpus availability
- cultural / domain bias
- cross-language fairness

---

## 10. 如果我们自己做 benchmark，建议的定位

一个更容易站住的方向是：

```text
面向真实多页文档的 evidence-grounded document QA benchmark，
专门评估模型从 noisy OCR / layout / tables / figures 中定位证据并完成可验证推理的能力。
```

这个定位的优点：

- 避免只做 OCR recognition。
- 避免只做普通 DocVQA。
- 可以自然覆盖 parser、retrieval、reasoning、hallucination。
- 可以设计 evidence-level annotation。
- 可以产生细粒度错误分析。

建议核心资产：

- 多页真实文档
- page-level / bbox-level evidence
- question type taxonomy
- answer type taxonomy
- reasoning type taxonomy
- OCR / layout noise annotation
- human-validated QA
- strong baseline suite
- error source decomposition

---

## 11. 最小可行版本设计

如果先做一个可投稿的 MVP，不建议一上来追求几十万样本。可以先做：

- 500-1,000 个高质量人工验证问题
- 100-300 份真实多页文档
- 5-8 类文档类型
- 6-10 类问题类型
- 每个问题带 evidence
- 每个问题至少双人验证
- 10-15 个强 baseline
- 充分 error analysis

关键是质量和诊断性，而不是样本数。

一个好的小 benchmark 也能中；一个大的但构建不透明的 benchmark 很容易被拒。

---

## 12. 最终建议

如果目标是做一篇 benchmark paper，建议把优先级排成：

1. 问题定义
2. 数据可信度
3. 标注和质检
4. 泄漏控制
5. 强 baseline
6. 诊断分析
7. 规模扩展
8. release 与复现

不要把主要精力都花在“收更多数据”上。benchmark 文章真正被 reviewer 反复拷问的是：

```text
我为什么要相信你的数据？
我为什么要相信你的评测？
我为什么要相信这个 benchmark 测到了一个新且重要的能力？
我为什么要相信你的结论不是由 bias、leakage、弱 baseline 或 metric artifact 导致的？
```

只要这四个问题能答硬，benchmark 才有可能从“数据整理工作”变成“定义一个可靠研究问题”的论文。

---

## 13. 参考 OpenReview 页面

### 中稿论文

- OCR-Reasoning Benchmark: https://openreview.net/forum?id=aH7eyx64pC
- DAVE: https://openreview.net/forum?id=kgk0NqjsoW
- TableMaster: https://openreview.net/forum?id=YyPZPrPjQD
- ChartGalaxy: https://openreview.net/forum?id=P4lFbvZ4HH
- Visual Self-Refine: https://openreview.net/forum?id=RI0oNr1b0y
- TABLET: https://openreview.net/forum?id=5UbeQDlYDj （补充候选；本轮未纳入逐篇 rebuttal 复盘）

### 拒稿论文

- HalluText: https://openreview.net/forum?id=LRnt6foJ3q
- End-to-End Document Understanding via Chain-of-Reading: https://openreview.net/forum?id=6YXMyPrDEN
- Structured Attention Matters: https://openreview.net/forum?id=3OnJAvuxd3
- GDI-Bench: https://openreview.net/forum?id=4l8QRqYzH9
- MDocAgent: https://openreview.net/forum?id=05SHW9ai9e
- CoTabBench: https://openreview.net/forum?id=wcInjlUp8V
- ChartNexus: https://openreview.net/forum?id=xg0fmtqh8d
- VisR-Bench: https://openreview.net/forum?id=7iFZ6uzILL
- DocPruner: https://openreview.net/forum?id=mEMGL1fLOO
