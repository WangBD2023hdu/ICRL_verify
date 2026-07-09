# 图像质量退化、拒答能力与解析幻觉率：文档解析方向调研报告

整理时间：2026-07-07  
调研范围：OpenReview 已验证材料、arXiv 论文、文档解析 / OCR / VQA / MLLM abstention / hallucination / document image quality 相关工作  
目标问题：在图像质量退化场景下，引入文档解析模型的拒答能力，并构建新的评价准则来评估解析幻觉率

---

## 1. 一句话结论

这个方向很值得做，而且比单纯做 OCR hallucination benchmark 更有方法和评测空间。但继续调研后，需要把边界说得更准：

```text
“退化 OCR + uncertainty tag + GRPO” 已经有 ICLR 2026 Poster 级别的强近邻；
“退化 KIE + refusal + hallucination-free accuracy” 已经有 NeurIPS 2025 工作；
因此新论文不能只停留在 OCR span 或字段 QA，
而要推进到结构化文档解析单元：token / field / cell / layout relation / chart value / evidence page。
```

核心机会是：

```text
现有文档解析 / OCR / DocQA benchmark 大多默认输入可读、问题可答；
现有 multimodal abstention 工作大多是 VQA / 多模态推理问题；
现有 hallucination benchmark 大多关注 object / caption / QA；
但真实文档解析系统最危险的失败是：
图像已经退化到不可可靠解析，模型仍然输出看似合理的字段、表格、布局或答案。
```

因此可以定义一个新问题：

```text
Degradation-aware selective document parsing:
当文档局部或整体不可读时，模型应当选择拒答、局部拒答或低置信输出，
而不是编造 OCR 文本、字段值、表格结构、布局关系或答案。
```

这件事的重点不只是“模型能不能回答”，而是：

- 哪些区域还可读？
- 哪些字段应该拒答？
- 哪些解析结果是 hallucinated？
- 模型是否知道自己不知道？
- 在不同退化强度下，模型的 hallucination rate 如何变化？
- 拒答带来的 coverage 损失和错误解析风险如何权衡？

最适合的论文定位：

```text
When Documents Become Unreadable:
Selective Refusal and Parsing Hallucination under Image Degradation
```

---

## 2. 相关论文地图

### 2.1 最直接相关：退化文档中的 OCR 幻觉与拒答

#### Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs

OpenReview：https://openreview.net/forum?id=zyCjizqOxB  
结果：ICLR 2026 Poster / Accept  
关键词：visually degraded document, uncertainty-aware OCR, GRPO, Blur-OCR

这是本方向目前最关键的 OpenReview 近邻。论文明确指出：VLM 正在替代传统 OCR，但在 lossy visual inputs，尤其是退化文档图像上，模型会输出流畅但错误的文本，而且不表达不确定性。作者提出 uncertainty-aware OCR：模型正常转写，但把不可靠 span 用 uncertainty tags 标出来。

方法上，它用 pseudo-labeled cold start + GRPO，设计了多目标 reward，同时平衡 transcription accuracy 和 uncertainty coverage，并加入机制防止 reward hacking。数据上，它提出 Blur-OCR benchmark，用于评估退化文档 OCR 中的 uncertainty tagging。OpenReview meta-review 认为 rebuttal 后主要问题基本解决：增加 backbone、补充 entropy / ensemble baseline、澄清 uncertainty 定义、加入 over-tagging / under-tagging 的 precision-recall 分析。最终三位 reviewer 保持正面，一位 reviewer 从 4 分改到 8 分。

对我们的启发：

- “退化文档 OCR 不应硬猜，而应表达不确定性”已经被主会接受。
- 指标不能只看 OCR accuracy，还要看 uncertainty tag precision / recall / F1。
- reward 设计必须防止两个极端：全部打 UNC tag 和完全不打 UNC tag。
- rebuttal 中 reviewers 很关注 synthetic degradation 是否能迁移到真实退化，这会是我们也必须防守的点。

对我们方向的约束：

- 如果只做 OCR span uncertainty，这篇已经占住主要贡献。
- 我们必须把任务扩展到 parsing units：字段、表格 cell、layout relation、chart value、multi-page evidence。
- 我们可以把 Blur-OCR 作为 OCR-level baseline/subtask，而不是主贡献终点。

结论：

这篇让方向更可信，但也压缩了 novelty。新的 paper 应该写成 “selective document parsing”，而不是 “uncertainty-aware OCR”。

---

#### Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models

链接：https://arxiv.org/abs/2506.20168  
结果：NeurIPS 2025 Accepted  
关键词：degraded document understanding, OCR hallucination, refusal, KIE-HVQA, GRPO

这是目前最贴近你想法的一篇工作。它指出：在真实文档场景中，视觉退化和歧义会让 MLLM 过度依赖语言先验，导致 OCR hallucination。作者提出 KIE-HVQA，用身份卡和发票等 KIE 场景构造退化输入，评估模型能否识别不可靠视觉信息并拒绝作答。

它的方法侧也很相关：作者在 Qwen2.5-VL 上做 SFT + GRPO，引入视觉不确定性的 self-awareness 和 refusal mechanism。论文报告其 7B 模型在 KIE-HVQA 上的 hallucination-free accuracy 相比 GPT-4o 有明显提升，同时标准任务没有显著退化。

对我们的启发：

- “图像退化 -> 视觉不确定 -> 应拒答而不拒答 -> OCR hallucination”这条问题链已经成立。
- refusal 不应该只是 prompt，而应该进入训练目标或 reward。
- 评价不能只看 accuracy，要看 hallucination-free accuracy。

局限：

- 任务主要是 KIE，文档类型偏身份卡和发票。
- 解析对象主要是字段级信息，不覆盖完整 layout、表格结构、chart、multi-page evidence。
- 没有把“解析幻觉率”系统扩展到 token / field / cell / structure / relation 多层级。

结论：

这篇是最重要的近邻工作。我们的新方向需要明确说明：不是重复 KIE-HVQA，而是把 refusal 和 hallucination 从 KIE 字段扩展到更一般的 document parsing。

#### Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR

链接：https://arxiv.org/abs/2504.11101  
关键词：OCR uncertainty, multi-VLM agreement, training-free verification

Consensus Entropy 提出一个训练无关的 OCR 可靠性估计方法：正确 OCR 输出在多个 VLM 之间更容易收敛，错误输出更容易分歧。它用多模型 agreement entropy 做 sample-level quality verification、best-output selection 和 adaptive routing。

对我们的启发：

- multi-model disagreement 是很强的不确定性 baseline，不能只和 prompt-based refusal 比。
- 对文档解析可以扩展成 multi-parser disagreement：OCR 文本、table HTML、layout blocks、reading order 如果跨系统分歧大，就触发局部 verification 或拒答。
- 这和 Dr. DocBench 的 parser-failure sampling 可以结合：训练/评估都优先关注 parser disagreement 高的区域。

---

### 2.2 OpenReview 相关：HalluText 的教训

#### HalluText: Towards Benchmarking and Mitigating OCR Hallucination for LVLMs

OpenReview：https://openreview.net/forum?id=LRnt6foJ3q  
结果：ICLR 2026 Reject

HalluText 试图系统评估和缓解 LVLM 的 OCR hallucination。方向本身很重要，但 OpenReview 反馈显示它被拒主要不是因为问题不重要，而是因为 benchmark 和方法证据链不够扎实：

- 数据构建细节不够清楚。
- QA 与 distractor 构造不够透明。
- 外部 OCR 依赖和 OCR 错误传播分析不足。
- mitigation 方法的必要性和替代 baseline 不够充分。
- 泛化范围偏 recognition / multiple-choice，较少覆盖真实 open-ended DocQA、layout、table、chart、reasoning。

对我们的启发：

如果要做“图像质量退化 + 拒答 + 解析幻觉率”，必须避免 HalluText 的风险：

- 不要只做选择题。
- 不要只看 OCR 字符串。
- 不要把数据构造写得像黑盒。
- 不要只证明模型会错，要定义更细的 hallucination taxonomy。
- 不要只靠一个 mitigation trick，要和 confidence、OCR score、DIQA、self-consistency、verifier 等强 baseline 比。

补充看 rebuttal 后，还有一个更具体的教训：作者在回复中补充了固定模板、规则生成 distractor、Position 类别的角度划分、双人标注、过滤规则、OCR 正确/错误子集分析、Top-1 OCR snippet baseline 等内容，但这些补充没有扭转最终 reject。这说明 benchmark paper 的关键质量控制不能主要靠 rebuttal 补，必须在主文中提前把数据生成、可读性判定、标注一致性、样本过滤、强 baseline 和失败案例分析写完整。

对我们尤其重要的是：如果设计 degraded parsing benchmark，必须把“什么时候该拒答”做成人类可验证标签，而不是只由 clean GT 和 degradation script 自动推出。

---

### 2.3 OpenReview 新近相关：低质量文档、恢复、结构一致性

#### DocRobust: Enhancing Robustness of Multi-modal LLMs in Low-Quality Document Image Scenarios

OpenReview：https://openreview.net/forum?id=lGfnvZsE2F  
结果：ICLR 2026 Withdrawn Submission  
关键词：low-quality document image, feature restoration, DocRobust-VQA, robustness

DocRobust 面向低质量文档图像理解，提出 DocRobust-Module，在 vision encoder 和 projector 之间做 feature-level restoration，并构建 DocRobust-VQA：约 189K clear-blurry image pairs 和 417K QA pairs。

审稿意见很有参考价值：

- 问题重要，低质量文档鲁棒性有实际需求。
- 方法新意偏弱，像常见 adapter / feature restoration。
- 只在 InternVL2.5 family 上验证，泛化不足。
- 退化类型只有 5 类，且具体实现、参数、可视例子不足。
- synthetic 和 real degradations 的 distribution shift 没分析。
- 严重退化时，clean image derived QA 可能在 degraded image 中已经不可验证，继续用原 GT 训练会引入噪声。

对我们的启发：

DocRobust 代表“把低质量图像恢复后继续回答”的路线。我们的路线应明确区分：恢复能提升可读性，但当证据仍不足时，系统应该拒答。换句话说，restoration baseline 需要纳入，但 refusal/PHR 才是安全可靠性的核心。

#### SAVIOR: Sample-efficient Alignment of Vision-Language Models for OCR Representation

OpenReview：https://openreview.net/forum?id=kiVIVBmMTP  
结果：ICLR 2026 Withdrawn Submission  
关键词：OCR representation, degraded scans, fine print, structure-aware metric

SAVIOR 关注企业文档中的 VLM-OCR 适配问题，强调真实失败案例包括 vertical text、stylized logo text、fine print、degraded scans 等。它构建了小规模高质量训练集和 financial documents benchmark，并提出 PAIRS，一种基于 token pair spatial relations 的 layout fidelity metric。

对我们的启发：

- 退化问题不只是 blur/compression，也包括 small text、fine print、logo/stylized text、扫描噪声。
- layout fidelity metric 可以启发 structure hallucination 的度量：不是只看文字对不对，还要看 token/field/cell 的空间关系是否被编造或错配。

#### GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?

OpenReview：https://openreview.net/forum?id=GxPtLwLSOL  
结果：ICLR 2026 Withdrawn Submission  
关键词：text image super-resolution, OCR fidelity, restoration

GLYPH-SR 做文本图像超分辨率，动机是普通 SR 指标如 PSNR、SSIM、LPIPS 对 character-level 错误不敏感，图像看起来更清楚不代表文字恢复正确。它用 VLM/OCR guidance 改善 text fidelity。

对我们的启发：

图像增强和超分辨率可以作为 degraded document pipeline 的上游模块，但它们本身也可能“美化并编造”文本。因此我们的 benchmark 应评估 restoration 后的 parsing hallucination，而不是只评估图像质量或 OCR F1。

#### Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription

OpenReview：https://openreview.net/forum?id=ybqL3FSOgG  
结果：ICLR 2026 Reject  
关键词：multi-page handwritten document transcription, HTR, OCR post-processing

这篇研究多页手写文档转录，提出 OCR+PAGE-1 / OCR+PAGE-N 等 prompting 策略，用单页视觉上下文帮助整个文档的 OCR post-processing。它被拒不是因为问题无意义，而是 reviewers 认为评估还不够多样：需要更强非 MLLM baseline、更长页数、更多脚本、成本分析和数据集深入分析。

对我们的启发：

- 多页文档里，模型会利用跨页 handwriting/style/context 去纠错，这对低质量文档很有价值。
- 但跨页上下文也可能放大语言先验，导致模型在不可读页面上“按上下文猜”。
- 我们可以专门测：跨页 context 是否降低错误，还是增加 unsupported completion。

#### Is Cognition Consistent with Perception? Assessing and Mitigating Multimodal Knowledge Conflicts in Document Understanding

OpenReview：https://openreview.net/forum?id=tBZK9BI2GZ  
结果：ICLR 2025 Withdrawn Submission  
关键词：document understanding, OCR-VQA consistency, cognition-perception conflict

这篇提出 C&P knowledge conflict：模型在 OCR/perception 上看到一个内容，但在 VQA/cognition 中回答另一个内容。它报告 GPT-4o 也只有约 68.6% C&P consistency，并提出 consistency fine-tuning。

对我们的启发：

Parsing hallucination 不只来自 OCR 看不清，也来自“感知结果”和“认知回答”之间不一致。我们的 verifier 可以要求每个结构化答案绑定视觉证据，避免模型在 OCR 正确的情况下仍然用先验或上下文给出不一致答案。

---

### 2.4 文档解析与 OCR benchmark

#### Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing

链接：https://arxiv.org/abs/2606.01393  
关键词：expert-level document parsing, difficulty-aware sampling, long documents, structural annotation

Dr. DocBench 是 2026 年新出现的困难文档解析 benchmark。它从大规模多语言书籍中选择长文档，覆盖 52 个 BISAC subject domains，平均文档长度约 100 页。关键设计是 parser-failure-based sampling：先跑多个强解析器，用跨解析器 disagreement 找出困难页面，再由人工和领域专家做标注。标注覆盖 layout、reading order、hierarchical relations、化学结构、乐谱、复杂表格、公式、算法伪代码等。

对我们的启发：

- 难样本不应只靠随机采样，而应主动选择当前 parser 分歧大的样本。
- 结构化解析评价应覆盖 text、formula、table、reading order 和 domain-specific visual content。
- 它没有直接做拒答/退化，但可以作为“专家级解析单元”的参考：我们的 PHR 不应只统计 OCR token，还应统计结构和专业符号是否 hallucinated。

#### PaddleOCR-VL-1.5 / Real5-OmniDocBench

链接：https://arxiv.org/abs/2601.21957  
关键词：robust in-the-wild document parsing, scanning, skew, warping, screen-photography, illumination

PaddleOCR-VL-1.5 报告在 OmniDocBench v1.5 上达到 94.5%，并提出 Real5-OmniDocBench，用于评估真实物理畸变下的文档解析鲁棒性，包括 scanning、skew、warping、screen-photography、illumination。

对我们的启发：

- 退化类型不应只停留在图像处理滤波；真实采集链路中的 skew、warping、屏幕拍摄、光照更重要。
- Real5-OmniDocBench 可作为真实退化类型设计参考，也可作为 baseline 系统评估对象。

#### MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

链接：https://arxiv.org/abs/2509.22186  
关键词：high-resolution document parsing, coarse-to-fine, layout-content decoupling

MinerU2.5 采用两阶段 coarse-to-fine 策略：低分辨率做全局 layout analysis，高分辨率 crop 做局部内容识别，从而兼顾效率和密集文本、公式、表格细节。

对我们的启发：

- 对退化文档，局部高分辨率 crop / reread 是非常强的 baseline。
- 如果局部 reread 后仍证据不足，系统才应局部拒答。这可以形成“repair-then-refuse”的方法框架。

#### ABot-OCR Technical Report

链接：https://arxiv.org/abs/2605.27978  
关键词：end-to-end document parsing, Markdown, structure-constrained RL, DPCS

ABot-OCR 直接把 page image 转成 clean Markdown，并用结构约束强化学习优化结构质量。它的数据引擎中有 Document Parsing Consistency Score (DPCS)，从 text fidelity、layout localization、reading order、structural fidelity、format validity、semantic completeness 等维度验证标注和伪标签是否与图像一致。

对我们的启发：

- DPCS 的维度很适合作为 Parsing Hallucination Verifier 的原型。
- 它强调 annotation verification，而我们的任务可以进一步把 verification 结果变成 refusal / hallucination 标签。
- 结构约束 RL 可以扩展为 refusal-aware structural RL：只有视觉证据足够时才奖励结构化输出，否则奖励局部拒答。

#### OCRBench

链接：https://arxiv.org/abs/2305.07895  
关键词：OCR, text recognition, document VQA, KIE, HMER

OCRBench 是早期系统评估 large multimodal models OCR 能力的 benchmark，覆盖 text recognition、scene text VQA、document-oriented VQA、KIE、handwritten mathematical expression recognition 等任务。

局限：

- 主要评估模型能不能识别与回答。
- 默认样本大多是可答的。
- 不强调拒答和图像退化下的不确定性。

#### OCRBench v2

链接：https://arxiv.org/abs/2501.00321  
关键词：visual text localization, reasoning, bilingual, difficult samples

OCRBench v2 扩展到 10,000 个 human-verified QA pairs，覆盖 31 种场景，包括 receipt、formula、diagram 等，并强调 text localization、handwritten extraction、layout perception、complex element parsing、logical reasoning。

对我们的启发：

- 它说明当前 LMM 在 fine-grained perception、layout perception、complex element parsing 上仍然弱。
- 但它仍主要是能力测试，不是“图像不可读时是否拒答”的选择性解析测试。

#### MMLongBench-Doc

链接：https://arxiv.org/abs/2407.01523  
关键词：long-context document understanding, multi-page, unanswerable questions

MMLongBench-Doc 包含 130 份长 PDF 文档和 1,062 个专家标注问题，平均 49.4 页。它的重要性在于：答案可能来自 text、image、chart、table、layout structure，并且 33.2% 是 cross-page question，22.8% 被设计为 unanswerable，用于检测 hallucination。

对我们的启发：

- 它已经把 unanswerable 引入长文档理解。
- 但它不是专门研究图像质量退化，也不是 document parsing 输出层面的拒答。
- 它可以作为我们构造 multi-page degradation refusal benchmark 的数据来源之一。

#### OmniDocBench

链接：https://arxiv.org/abs/2412.07626  
关键词：PDF document parsing, layout categories, diverse document types

OmniDocBench 是 PDF document parsing 方向的重要 benchmark，覆盖九类文档，提供 19 个 layout category labels 和 14 个 attribute labels，可以评估 modular pipelines 和 end-to-end methods。

对我们的启发：

- 它适合作为 document parsing 结构评估基础。
- 但它主要评估解析质量，不强调低质量图像时的拒答能力。
- 可以借用其 layout category / attribute label 设计，扩展出 “parseable / unparseable / hallucinated” 标签。

---

### 2.5 多模态拒答与 unanswerable VQA

#### Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly

链接：https://arxiv.org/abs/2204.13631  
关键词：VQA abstention, risk-coverage, effective reliability

这篇较早把 abstention 引入 VQA。它把 VQA 变成 selective prediction：模型可以选择回答或拒答。论文用 coverage 和 risk 描述权衡，并提出 Effective Reliability metric。

对我们的启发：

- 文档解析也应该引入 risk-coverage curve。
- 不能只看拒答率，必须看回答覆盖率和非拒答样本上的错误率。
- 对错误解析应赋予比拒答更高的成本。

#### VizWiz

链接：https://arxiv.org/abs/1802.08218  
关键词：blind users, poor image quality, unanswerable questions

VizWiz 来自真实盲人用户拍摄图像和语音问题。它的重要性在于：图像经常质量差，问题也经常不可答。它是“真实低质量图像 + 应拒答问题”的经典来源。

对我们的启发：

- 真实用户数据中的不可答不是人工设定，而是自然出现。
- 文档解析也可以收集真实低质量扫描、手机拍摄、反光、遮挡、模糊文档。
- 人类标注可以包括 “cannot determine from image”。

#### UNK-VQA

链接：https://arxiv.org/abs/2310.10942  
关键词：unanswerable VQA, multimodal large models, perturbation

UNK-VQA 通过对图像或问题做语义接近的扰动来构造 unanswerable questions，并评估多模态大模型的 abstention 能力。

对我们的启发：

- 构造 unanswerable 样本时，不能只简单换图或遮掉整张图，否则太容易。
- 应保持样本接近原分布，让拒答判断变难。
- 文档退化样本也应该避免太粗暴，比如不是简单全黑，而是局部字段模糊、表格列压缩、低分辨率小字不可辨。

#### MoHoBench

链接：https://arxiv.org/abs/2507.21503  
关键词：MLLM honesty, unanswerable visual questions, refusal

MoHoBench 用 12k+ unanswerable visual question samples 评估 MLLM 的 honesty。它发现很多模型在必要时不能恰当拒答，并指出 multimodal honesty 不只是语言模型问题，而深受视觉信息影响。

对我们的启发：

- 拒答能力需要专门对齐，而不是靠通用 instruction following。
- 文档解析中的 honesty 可以定义为：不输出没有视觉证据支持的字段/结构/答案。

#### Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems

链接：https://arxiv.org/abs/2604.14799  
关键词：MM-AQA, multimodal abstention, evidence insufficiency, MMLongBench-Doc

这篇提出 MM-AQA，从 MMMU 和 MMLongBench-Doc 构造 unanswerable instances，用 visual modality dependency 和 evidence sufficiency 两条轴设计 transformations。它发现标准 prompting 下 VLM 很少主动 abstain，confidence baseline 甚至能超过直接 prompt；multi-agent system 能改善 abstention，但会带来 accuracy-abstention trade-off；模型面对 degraded 或 contradictory evidence 时常试图调和而不是拒答。

对我们的启发：

- 退化证据比“证据完全缺失”更难，因为模型会试图猜。
- benchmark 构造应按 evidence dependency 做 question-aware transformation。
- 评价必须用五分类或多分类 confusion matrix，而不是简单 accuracy。
- 这篇是我们设计文档退化拒答评价框架的重要参考。

#### TUBench: Benchmarking Large Vision-Language Models on Trustworthiness with Unanswerable Questions

链接：https://arxiv.org/abs/2410.04107  
关键词：unanswerable questions, trustworthiness, statistical tables, code screenshots

TUBench 用十种策略构造不可答问题，覆盖 code screenshots、natural images、geometry diagrams 和 statistical tables。虽然不是文档解析 benchmark，但它把“表格/截图上的不可答问题”纳入多模态 trustworthiness，很接近文档 QA 中的 missing evidence / insufficient evidence。

对我们的启发：

- 不可答样本不应只有遮挡，还可以来自问题本身要求的信息不存在、证据不足或约束缺失。
- 对表格和图表页面，问题设计应区分“数值不可读”“列不存在”“单位缺失”“需要外部知识”。

---

### 2.6 文档图像质量评价

#### DocIQ

链接：https://arxiv.org/abs/2509.17012  
关键词：document image quality assessment, DIQA-5000, quality score

DocIQ 提出 DIQA-5000，包含 5,000 个 document images，并让人类从 overall quality、sharpness、color fidelity 三个维度打分。它还提出 no-reference DIQA 模型，结合 document layout features 做质量预测。

对我们的启发：

- 图像质量评分可以作为拒答 baseline。
- 但 DIQA 本身只预测质量，不判断某个字段/表格/区域是否可解析。
- 我们需要把 document quality score 转成 parseability / answerability，而不是只做主观质量评价。

#### DeQA-Doc

链接：https://arxiv.org/abs/2507.12796  
关键词：MLLM-based document image quality assessment

DeQA-Doc 将 MLLM-based image quality scoring 扩展到文档质量评价，支持多种退化类型并输出连续质量分数。

对我们的启发：

- MLLM 可以作为 document quality assessor。
- 可作为模型拒答前的 quality gate。
- 但需要避免 quality score 与解析正确性脱钩：有些图像整体质量低但关键字段可读；有些整体清晰但某个字段被遮挡。

#### CG-DIQA / text-line based DIQA

链接：https://arxiv.org/abs/1807.04047  
链接：https://arxiv.org/abs/1906.01907

这些工作用 character gradient、text line detection、OCR accuracy proxy 等方法评估文档图像质量。

对我们的启发：

- 字符/文本行级质量比整图质量更适合文档解析。
- 我们可以设计 field-level / line-level / cell-level parseability score。

---

### 2.7 视觉幻觉评价

#### POPE

链接：https://arxiv.org/abs/2305.10355  
关键词：object hallucination, polling-based probing

POPE 用 polling-based query 评估 LVLM 是否生成图像中不存在的对象。它的核心思想是：通过有控制的 yes/no probing 检测模型是否声称看到了不存在的东西。

对我们的启发：

- 文档解析也可以做 polling-style hallucination probing。
- 例如问模型某个字段、表格行、页码、图例、印章是否存在。
- 对不存在字段的肯定回答就是解析幻觉。

#### HallusionBench

链接：https://arxiv.org/abs/2310.14566  
关键词：language hallucination, visual illusion, controlled question pairs

HallusionBench 通过专家构造图像和问题，分析模型的 response tendency、logical consistency 和 failure modes。

对我们的启发：

- 幻觉评价最好有 control groups。
- 文档退化 benchmark 可以构造 clean / degraded / contradictory / masked 成对样本，分析模型是否一致。

#### A Survey on Hallucination in Large Vision-Language Models

链接：https://arxiv.org/abs/2402.00253

该综述系统梳理了 LVLM hallucination 的症状、评价 benchmark、原因和缓解方法。

对我们的启发：

- 视觉幻觉的已有 taxonomy 主要围绕 object / attribute / relation。
- 文档解析需要新的 hallucination taxonomy：text hallucination、field hallucination、layout hallucination、table structure hallucination、chart value hallucination、provenance hallucination。

---

## 3. 现有工作的空白

### 3.0 新边界：OCR span uncertainty 已经不够新

继续调研后，必须承认一个事实：如果论文只说“模糊 OCR 时给不确定 span 打标签”，很容易被认为和 ICLR 2026 Poster 的 uncertainty-aware OCR 重合。如果论文只说“低质量 KIE 时拒答”，又容易被认为和 KIE-HVQA / Seeing is Believing 重合。

真正的空白应该重新表述为：

```text
从 OCR / KIE 的 answer-or-refuse，
扩展到完整 document parsing 的 parse-or-refuse；
从 span uncertainty，
扩展到结构化解析单元的证据充分性；
从 hallucination-free QA accuracy，
扩展到 token / field / cell / layout / relation / chart / evidence 的 Parsing Hallucination Rate。
```

也就是说，核心 novelty 不在“拒答”两个字，而在“文档解析单元级拒答 + 结构化幻觉率 + 证据绑定评价”。

### 3.1 文档解析 benchmark 很少要求模型拒答

OCRBench、OCRBench v2、OmniDocBench 主要衡量能不能识别、定位、解析。MMLongBench-Doc 虽然有 unanswerable questions，但重点是长文档理解，不是图像质量退化下的 selective parsing。

缺口：

- 缺少对 degraded document image 的系统评估。
- 缺少 parseability / answerability 标签。
- 缺少“局部拒答”评价。
- 缺少非拒答输出的 hallucination rate。

### 3.2 VQA abstention 不是 document parsing abstention

VQA 拒答通常是对整个问题输出 “I don’t know”。但文档解析更复杂：

- 整页不可读；
- 某个字段不可读；
- 某个 table cell 不可读；
- 某个 chart value 不可读；
- layout 可读但文本不可读；
- 文本可读但表格结构不可确定；
- 多个视觉解释都可能成立。

因此文档解析需要更细粒度的拒答：

- document-level refusal
- page-level refusal
- region-level refusal
- field-level refusal
- cell-level refusal
- relation-level refusal
- answer-level refusal

### 3.3 图像质量评价不能直接等价于解析可靠性

DIQA 可以预测图像质量，但质量高低不等于某个任务是否可答：

- 图像整体模糊，但目标字段很大，仍可读。
- 图像整体清晰，但关键字段被遮挡，应该拒答。
- 表格文字可读，但行列线缺失，结构不可确定。
- chart 图像清晰，但 legend 被裁掉，数值不可推断。

因此需要 task-aware / evidence-aware quality evaluation。

### 3.4 幻觉评价需要从 object 扩展到 parsing units

文档解析中的 hallucination 不是“看到不存在的 object”，而是：

- 编造不存在的文字；
- 把模糊字段猜成常见值；
- 补全缺失数字；
- 生成不存在的表格行/列；
- 错配 key-value；
- 错配 figure-caption；
- 编造 chart 数值；
- 引用错误页面作为证据；
- 输出没有视觉 evidence 支撑的答案。

这需要新的 Parsing Hallucination Rate。

---

## 4. 建议的新任务定义

### 4.1 任务名称

可以叫：

```text
Degradation-Aware Selective Document Parsing
```

或者：

```text
Selective Document Parsing under Visual Degradation
```

### 4.2 输入

输入包括：

- document image 或 multi-page PDF screenshots
- task instruction，例如 OCR / KIE / table parsing / layout parsing / DocQA
- optional parser / OCR outputs

图像包含不同质量退化：

- low resolution
- Gaussian blur
- motion blur
- JPEG compression
- noise
- under/over exposure
- shadow / glare
- occlusion
- crop
- skew / perspective distortion
- watermark
- bleed-through
- scan artifacts
- small text
- overlapping text
- handwritten noise

### 4.3 输出

模型不应该被迫总是输出解析结果，而应输出：

```json
{
  "status": "answer" | "partial_answer" | "refuse",
  "parse": {},
  "uncertain_regions": [],
  "refusal_reason": "blurred_text | occluded_field | ambiguous_layout | missing_evidence | low_resolution | contradictory_evidence",
  "confidence": 0.0,
  "evidence": []
}
```

对于 KIE：

```json
{
  "name": {"value": "张三", "status": "answer", "confidence": 0.94},
  "id_number": {"value": null, "status": "refuse", "reason": "motion_blur"},
  "date": {"value": "2024-05-12", "status": "answer", "confidence": 0.88}
}
```

对于 table parsing：

```json
{
  "cells": [
    {"row": 1, "col": 2, "text": "Revenue", "status": "answer"},
    {"row": 3, "col": 4, "text": null, "status": "refuse", "reason": "occluded_cell"}
  ],
  "structure_status": "partial_answer"
}
```

### 4.4 标签

每个样本最好有多层标签：

- clean ground truth
- degraded image
- degradation type
- degradation severity
- human parseability label
- answerability label
- element-level visible / invisible
- acceptable refusal regions
- hallucination-critical regions

标签级别：

- page-level
- region-level
- field-level
- token-level
- table-cell-level
- layout-relation-level

---

## 5. 新评价准则：Parsing Hallucination Rate

### 5.1 核心定义

解析幻觉率衡量：

```text
模型在视觉证据不足或不可读时，仍然输出了无视觉支持的解析内容的比例。
```

一个基础版本：

```text
PHR = hallucinated_parse_units / non_refused_parse_units
```

其中 parse units 可以是：

- OCR token
- KIE field
- table cell
- layout block
- key-value relation
- chart mark/value
- page evidence
- final answer

### 5.2 不同层级的 PHR

#### Token Hallucination Rate

```text
THR = unsupported_tokens / generated_tokens
```

适合 OCR / text extraction。

#### Field Hallucination Rate

```text
FHR = fabricated_or_wrong_fields / answered_fields
```

适合 KIE / form understanding。

#### Cell Hallucination Rate

```text
CHR = hallucinated_cells / answered_cells
```

适合 table parsing。

#### Structure Hallucination Rate

```text
SHR = hallucinated_structure_relations / predicted_structure_relations
```

适合 layout / table structure / reading order。

#### Evidence Hallucination Rate

```text
EHR = unsupported_answers / non_refused_answers
```

适合 DocQA / multi-page QA。

### 5.3 拒答相关指标

#### Refusal Precision

模型拒答的样本中，确实不可可靠解析的比例。

```text
RP = correct_refusals / all_refusals
```

#### Refusal Recall

所有应该拒答的样本中，模型成功拒答的比例。

```text
RR = correct_refusals / all_unparseable_units
```

#### Over-Refusal Rate

可解析内容被模型错误拒答的比例。

```text
ORR = wrong_refusals_on_parseable_units / all_parseable_units
```

#### Under-Refusal Hallucination Rate

应该拒答但模型给出错误解析的比例。

```text
UHR = hallucinated_answers_on_unparseable_units / all_unparseable_units
```

### 5.4 Selective Parsing Risk-Coverage Curve

借鉴 selective classification 和 Reliable VQA：

- coverage：模型选择回答/解析的比例。
- risk：在被回答/解析样本上的错误率或 hallucination rate。

理想模型：

- 在高质量图像上 coverage 高、risk 低；
- 在低质量图像上主动降低 coverage；
- risk 不应随退化严重程度急剧上升。

### 5.5 Hallucination-Free Accuracy

可以定义：

```text
HFA = (correct_answers + correct_refusals) / all_units
```

但要注意：

- HFA 容易鼓励过度拒答。
- 所以必须同时报告 coverage 和 over-refusal。

更好的主指标可以是：

```text
Selective Utility = correct_answer_reward
                  - hallucination_penalty
                  - over_refusal_penalty
                  - unnecessary_partial_refusal_penalty
```

其中 hallucination_penalty 应显著高于 refusal_penalty。

### 5.6 Parseability Confusion Matrix

建议不要只用二分类 accuracy，而是为每个 parse unit 建一个混淆矩阵：

```text
GT parseable + model answer correct      = supported correct
GT parseable + model answer wrong        = ordinary parsing error
GT parseable + model refuse              = over-refusal
GT unparseable + model refuse            = correct refusal
GT unparseable + model answer            = parsing hallucination
GT ambiguous + model commits one answer  = ambiguity hallucination
GT ambiguous + model marks uncertain     = calibrated uncertainty
```

这个矩阵比单个 hallucination-free accuracy 更稳，因为它能区分：

- 模型解析能力差；
- 模型不会拒答；
- 模型过度拒答；
- 模型面对歧义时硬猜；
- 模型能输出部分解析但保留不确定单元。

### 5.7 Uncertainty Tag Metrics

借鉴 uncertainty-aware OCR，还应报告：

- uncertainty tag precision：被标为不确定的单元是否真的不可可靠解析；
- uncertainty tag recall：不可可靠解析的单元是否被标出；
- uncertainty tag F1；
- over-tagging rate；
- under-tagging hallucination rate。

但我们的 unit 不应只限于文字 span，而应包含：

- OCR token / line；
- KIE field；
- table cell；
- row-column relation；
- layout block；
- figure-caption relation；
- chart value / legend；
- multi-page evidence span。

---

## 6. 数据构建建议

### 6.1 paired clean-degraded 设计

每个样本最好有：

- clean image
- clean ground truth
- degraded image
- degradation type
- severity
- human parseability decision

好处：

- clean ground truth 提供可验证答案。
- degraded image 决定是否仍可读。
- 可以画 degradation sensitivity curve。

### 6.2 合成退化 + 真实退化

合成退化用于控制变量：

- blur severity
- compression quality
- downsampling ratio
- occlusion size
- crop ratio
- perspective angle

真实退化用于生态有效性：

- 手机拍摄发票/合同/证件；
- 反光、阴影、抖动；
- 旧扫描件；
- 传真/复印；
- 低质量 PDF 截图；
- 拍摄角度不正；
- 表格线断裂。

### 6.3 局部退化比整体退化更重要

不要只把整张图变糊。更真实的情况是：

- 只有身份证号模糊；
- 只有表格右下角被遮挡；
- 只有 chart legend 被裁掉；
- 只有一页里的一段小字不可读；
- 只有某个 field 被印章覆盖。

局部退化能迫使模型做 partial refusal，而不是整图拒答。

### 6.4 标注协议

建议标注者对每个 unit 标：

- readable
- partially readable
- unreadable
- ambiguous
- missing
- contradictory

并要求：

- 至少双人标注；
- 记录 disagreement；
- 计算 inter-annotator agreement；
- 对 ambiguous case 给规则；
- 对 partial readable 给可接受答案集合。

### 6.5 Difficulty-aware / disagreement-aware sampling

建议不要完全随机采样 degraded pages。可以借鉴 Dr. DocBench 和 Consensus Entropy：

- 用多个 OCR / parser / VLM 跑 clean 和 degraded image；
- 计算跨系统 disagreement；
- 优先选择 disagreement 高、但人类仍能判断 parseability 的样本；
- 对 clean 可解析但 degraded 不可解析的区域做 hard negative；
- 对 parser 输出一致但实际错误的区域做 hallucination trap。

这样能避免 benchmark 太容易，也能解释为什么你的数据集能发现现有 benchmark 看不到的风险。

### 6.6 真实退化必须有“可验证性”标注

DocRobust 的审稿意见提醒了一个关键问题：如果 QA 是从 clean image 派生的，严重退化后原答案可能已经在 degraded image 中不可验证。此时继续把 clean GT 当唯一答案，会把“应该拒答”的样本错误标成“模型没答对”。

所以每个 degraded sample 都需要单独标：

- 从 degraded image 本身是否可读；
- 人类是否能在不看 clean image 的情况下给出答案；
- 若看不清，允许哪些形式的 partial answer；
- 若多个答案都合理，是否应标为 ambiguous；
- clean GT 只作为 reference，不直接等于 degraded GT。

---

## 7. Baseline 设计

### 7.1 基础模型

- Direct MLLM
- MLLM + “如果看不清请拒答” prompt
- MLLM + verbal confidence
- MLLM + CoT
- MLLM + self-consistency
- MLLM + self-refine

### 7.2 OCR / parser baseline

- OCR confidence threshold
- OCR ensemble disagreement
- parser confidence threshold
- OCR + LLM with refusal prompt
- layout parser + LLM
- table parser + LLM
- PaddleOCR-VL / MinerU2.5 / ABot-OCR / other strong document parsers
- high-resolution crop reread baseline
- parser-disagreement selective abstention

### 7.3 图像质量 baseline

- DIQA score threshold
- text-line quality threshold
- field-region quality threshold
- DeQA-Doc-style MLLM quality score

### 7.4 不确定性 baseline

- softmax / logprob confidence if available
- verbal confidence
- entropy / semantic entropy
- multi-sample disagreement
- multi-model disagreement
- Consensus Entropy / CE-OCR style agreement
- verifier score

### 7.5 方法型 baseline

- KIE-HVQA / GRPO-style refusal training
- uncertainty-aware OCR / Blur-OCR style UNC tagging
- MM-AQA-style verifier
- DocRobust-style feature restoration
- DocRes / PreP-OCR / super-resolution restoration before parsing
- visual self-verification
- evidence rendering + local reread
- cost-aware selective parser

### 7.6 强 baseline 的审稿意义

这里要特别注意：HalluText、DocRobust、Judge-a-Book 这些 OpenReview 案例都被 reviewers 追问过 baseline 是否充分。我们的实验至少需要覆盖四类强对照：

- always answer：直接解析，看 hallucination rate；
- prompt refuse：只靠提示词拒答；
- confidence gate：OCR confidence / DIQA / entropy / CE；
- repair then answer：restoration / crop reread / parser ensemble；
- train to refuse：SFT/RL/refusal-aware training。

如果我们的方法只比 direct prompt 好，不够；必须说明它比“质量门控 + 强 parser + 局部重读”更稳，或者在相同 coverage 下 PHR 更低。

---

## 8. 可以做的方法创新

### 8.1 Quality-Aware Evidence Verifier

模型先输出解析结果，再由 verifier 判断每个结果是否有足够视觉证据支持。

Verifier 输入：

- degraded image crop
- predicted parse unit
- OCR confidence
- quality score
- evidence bbox

Verifier 输出：

- supported
- unsupported
- uncertain
- refuse

优势：

- 不强迫生成模型自己判断所有不确定性。
- 可以做 element-level refusal。

### 8.2 Visual Evidence Rendering

把模型解析出的字段/单元格/布局关系画回图像上，让模型二次检查：

- 这个 bbox 里是否真的有这个文字？
- 这个 cell 是否真的存在？
- 这个 key-value 是否匹配？
- 这个 chart value 是否从图上可读？

优势：

- 和 Visual Self-Refine 思路一致，但扩展到文档解析。
- 能直接减少 unsupported outputs。

### 8.3 Local Repair Instead of Global Retry

如果 verifier 判定某个字段不可读，不要重跑整页，而是：

- high-res crop
- deblur / super-resolution
- OCR rerun
- ask model to compare multiple candidates
- 若仍不确定则局部拒答

优势：

- 控制成本。
- 比全局 self-refine 更可解释。

### 8.4 Refusal-Aware Training / RL

训练目标中加入：

- 正确解析奖励；
- 正确拒答奖励；
- 幻觉惩罚；
- 过度拒答轻惩罚；
- partial answer 奖励。

可以借鉴 KIE-HVQA 的 GRPO/refusal reward 思路，但扩展到多粒度 parsing units。

### 8.5 Parsing-Unit Evidence Binding

每个输出单元都强制绑定 evidence：

```json
{
  "unit": "table_cell",
  "value": "12.7%",
  "bbox": [120, 340, 210, 365],
  "page": 3,
  "status": "answer",
  "evidence_quality": "readable"
}
```

如果模型不能给出 evidence bbox / page / crop，或者 verifier 判断 evidence 不支持 value，则该输出进入 hallucination 候选。这个设计可以把“结构化解析结果是否有视觉证据”变成可审计对象。

### 8.6 Repair-Then-Refuse Pipeline

一个更像真实系统的方法框架：

```text
initial parse
-> unit-level verifier
-> if uncertain: crop / enhance / high-res reread / parser ensemble
-> if still uncertain: local refusal
-> output partial parse with uncertainty map
```

这比“直接拒答”更容易被接受，因为它先尽力恢复可读信息，再对仍不可验证的部分拒答；同时也比“盲目 restoration 后继续回答”更安全。

### 8.7 Structure-Aware Refusal Reward

借鉴 ABot-OCR 的 DPCS 和 structure-constrained RL，可以把 reward 拆成：

- content accuracy reward；
- layout / reading-order reward；
- table/formula structure reward；
- evidence support reward；
- correct refusal reward；
- hallucination penalty。

关键是不要让 reward 鼓励模型通过过度拒答拿高分，也不要让模型为了结构完整性补全不存在的行、列、公式或字段。

---

## 9. 推荐论文定位

### 9.1 题目方向

```text
When Documents Become Unreadable:
Selective Refusal and Parsing Hallucination under Image Degradation
```

或者：

```text
DegradeDoc: Evaluating and Mitigating Parsing Hallucinations in Degraded Document Understanding
```

### 9.2 核心贡献

可以写成四点：

1. 提出 degraded document selective parsing 任务，要求模型在低质量图像下做局部/整体拒答。
2. 构建包含 clean-degraded pairs、element-level parseability labels 和多级退化类型的数据集。
3. 提出 Parsing Hallucination Rate、Selective Parsing Risk-Coverage、Hallucination-Free Accuracy 等评价准则。
4. 提出 quality-aware evidence verifier / visual evidence rendering / refusal-aware training 方法，降低解析幻觉。

### 9.3 和已有工作的差异

相对 uncertainty-aware OCR / Blur-OCR：

- 从 OCR span uncertainty 扩展到 document parsing unit uncertainty。
- 从转写文本中的 UNC tag 扩展到字段、表格 cell、layout relation、chart value、证据页。
- 从 uncertainty tag F1 扩展到 PHR、parseability confusion matrix 和 structure-aware selective utility。
- 从 synthetic blur OCR 扩展到真实采集退化和结构化解析退化。

相对 KIE-HVQA：

- 从 KIE 扩展到 OCR、layout、table、chart、DocQA。
- 从 field-level refusal 扩展到 token/cell/region/relation 多层级 refusal。
- 从 hallucination-free accuracy 扩展到 PHR 和 risk-coverage。

相对 MM-AQA / VQA abstention：

- 从 whole-question abstention 扩展到 document parsing unit abstention。
- 从 general multimodal reasoning 扩展到文档结构、OCR、layout、table。

相对 OCRBench / OmniDocBench：

- 从“能否解析”扩展到“何时不应解析”。
- 从 accuracy 扩展到 selective reliability。

相对 HalluText：

- 不只看 OCR hallucination，还看 parsing hallucination。
- 不只做 multiple-choice，而是结构化输出。
- 更强调数据构建透明、人工可读性标注和强 baseline。

---

## 10. 审稿风险与防御

### 风险 1：这是不是只是 uncertainty-aware OCR / KIE-HVQA 的扩展？

防御：

- 明确任务更广：OCR、layout、table、chart、DocQA、multi-page evidence。
- 指标更细：token/field/cell/structure/evidence PHR，不只是 UNC-tag F1 或 hallucination-free QA accuracy。
- 输出更复杂：partial parsing + local refusal。
- 实验上必须包含 Blur-OCR-like OCR 子任务和 KIE-HVQA-like 字段子任务作为对照，而主贡献放在结构化解析单元。

### 风险 2：退化是合成的，不真实

防御：

- 同时收真实退化数据。
- 合成退化只用于控制变量。
- 报 real-only subset 和 synthetic-to-real transfer。
- 用 DIQA / human quality score 证明退化分布合理。

### 风险 3：拒答会被模型滥用

防御：

- 报 over-refusal rate。
- 报 coverage。
- 用 Selective Utility 惩罚过度拒答。
- 对 answerable clean subset 保证性能不下降。

### 风险 4：LLM judge 不可靠

防御：

- 优先用结构化 ground truth 和规则评价。
- LLM judge 只用于开放式解释。
- 抽样做人类验证和 judge agreement。

### 风险 5：benchmark paper 可信度不足

防御：

- 详细写数据来源、退化生成、人工标注、IAA、过滤规则。
- 开源 clean/degraded pairs、标注工具、evaluation script。
- 提供 baseline code。

### 风险 6：这是不是 restoration / robust parsing 就能解决？

防御：

- 把 DocRobust、DocRes、PreP-OCR、super-resolution、high-res crop reread 都作为 repair baseline。
- 报告 repair 成功后的 coverage 提升，也报告 repair 失败但模型继续回答时的 PHR。
- 强调目标不是替代 restoration，而是决定 restoration 后哪些单元仍然证据不足。
- 用严重局部遮挡、关键字段缺失、chart legend 被裁掉等样本证明“增强图像”不能凭空恢复证据。

---

## 11. 最终建议

这个方向最好不要只写成：

```text
我们构建了一个退化文档 OCR hallucination benchmark。
```

更强的写法是：

```text
我们提出 selective document parsing under visual degradation，
把文档解析从 always-answer 模式改成 answer-or-refuse 模式；
并提出 Parsing Hallucination Rate 来衡量模型在不可读区域编造解析结果的风险。
```

最关键的创新点是：

```text
拒答不是整题拒答，而是文档解析单元级拒答；
幻觉不是答案错，而是结构化解析结果缺少视觉证据。
```

这会把论文从普通 benchmark 推到一个更有研究味道的问题：

- 文档质量退化；
- 视觉不确定性；
- 选择性解析；
- 局部拒答；
- 解析幻觉；
- 风险-覆盖权衡；
- evidence-aware verification。

如果后续要继续推进，我建议先做一个小型 MVP：

- 3 类文档：证件/发票、表格报告、图表页面；
- 5 类退化：blur、low-res、compression、occlusion、crop；
- 每类 200-300 个 clean-degraded pairs；
- 标 field/token/cell/region parseability；
- 跑 6-8 个 MLLM + OCR/parser baseline；
- 加入至少 3 类强 baseline：uncertainty-aware OCR/entropy/CE，restoration + parser，high-res crop reread；
- 报告 PHR、coverage、over-refusal、uncertainty tag F1、parseability confusion matrix；
- 先验证 PHR 是否能揭示 accuracy 看不到的问题。

只要这个 MVP 能证明“模型在低质量图像下不是简单变差，而是系统性编造”，这个方向就有很强的论文潜力。

---

## 12. 参考文献与链接

### 退化文档与 OCR hallucination

- Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs: https://openreview.net/forum?id=zyCjizqOxB
- Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models: https://arxiv.org/abs/2506.20168
- HalluText: Towards Benchmarking and Mitigating OCR Hallucination for LVLMs: https://openreview.net/forum?id=LRnt6foJ3q
- Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR: https://arxiv.org/abs/2504.11101
- GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?: https://openreview.net/forum?id=GxPtLwLSOL
- SAVIOR: Sample-efficient Alignment of Vision-Language Models for OCR Representation: https://openreview.net/forum?id=kiVIVBmMTP
- DocRobust: Enhancing Robustness of Multi-modal LLMs in Low-Quality Document Image Scenarios: https://openreview.net/forum?id=lGfnvZsE2F

### OCR / 文档解析 benchmark

- Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing: https://arxiv.org/abs/2606.01393
- PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing: https://arxiv.org/abs/2601.21957
- MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing: https://arxiv.org/abs/2509.22186
- ABot-OCR Technical Report: https://arxiv.org/abs/2605.27978
- OCRBench: https://arxiv.org/abs/2305.07895
- OCRBench v2: https://arxiv.org/abs/2501.00321
- MMLongBench-Doc: https://arxiv.org/abs/2407.01523
- OmniDocBench: https://arxiv.org/abs/2412.07626
- Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription: https://openreview.net/forum?id=ybqL3FSOgG
- Is Cognition Consistent with Perception? Assessing and Mitigating Multimodal Knowledge Conflicts in Document Understanding: https://openreview.net/forum?id=tBZK9BI2GZ

### 多模态拒答 / unanswerable VQA

- Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly: https://arxiv.org/abs/2204.13631
- VizWiz Grand Challenge: https://arxiv.org/abs/1802.08218
- UNK-VQA: https://arxiv.org/abs/2310.10942
- MoHoBench: https://arxiv.org/abs/2507.21503
- Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems: https://arxiv.org/abs/2604.14799
- TUBench: Benchmarking Large Vision-Language Models on Trustworthiness with Unanswerable Questions: https://arxiv.org/abs/2410.04107

### 文档图像质量评价

- DocIQ: https://arxiv.org/abs/2509.17012
- DeQA-Doc: https://arxiv.org/abs/2507.12796
- CG-DIQA: https://arxiv.org/abs/1807.04047
- Text-line based DIQA: https://arxiv.org/abs/1906.01907

### 视觉幻觉评价

- POPE: https://arxiv.org/abs/2305.10355
- HallusionBench: https://arxiv.org/abs/2310.14566
- A Survey on Hallucination in Large Vision-Language Models: https://arxiv.org/abs/2402.00253

### 选择性预测与校准

- Selective Classification for Deep Neural Networks: https://arxiv.org/abs/1705.08500
- SelectiveNet: https://arxiv.org/abs/1901.09192
- On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
