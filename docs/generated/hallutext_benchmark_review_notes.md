# HalluText Review 复盘与 Benchmark 设计参考

整理日期：2026-07-07  
来源：OpenReview 论文页与本次讨论  
OpenReview 链接：https://openreview.net/forum?id=LRnt6foJ3q

## 1. 一句话结论

HalluText 不是因为方向没有价值被拒，而是因为审稿人认为它作为 benchmark 和方法论文的可信度链条还不够完整：数据构建细节不够透明，任务边界不够清楚，distractor 和标注协议不够可验证，方法实验比较与泛化验证也不够充分。

对想做 benchmark 的启发是：benchmark 论文的核心不是“我收集了一批题”，而是要证明这批题确实在测你声称的能力，而且别人可以相信、复现、比较和诊断模型。

## 2. HalluText 被拒稿的主要原因

论文：HalluText: Towards Benchmarking and Mitigating OCR Hallucination for LVLMs  
决定：Reject  
可见评审倾向：

| Reviewer | Rating | 核心态度 |
| --- | --- | --- |
| 4VgK | 2 | 明确 reject |
| VCNM | 4 | marginally below acceptance |
| TEL7 | 4 | marginally below；rebuttal 后表示愿意提高分 |
| ne8t | 4 | marginally below；confidence 高，批评最系统 |

Area Chair 的 meta-review 总结为：benchmark design inadequacies、methodological limitations、experimental gaps、insufficient comparisons、presentation issues。

### 2.1 Benchmark 描述不够扎实

Reviewer 4VgK 和 ne8t 都指出，HalluText 作为 benchmark paper，数据构建过程写得不够细。主要问题包括：

- 数据清洗和过滤标准不够具体。
- QA pair 如何生成没有充分说明。
- distractor 如何保证“有挑战但不歧义”没有充分论证。
- Position 类任务中相邻类别的边界不够清晰，例如 top-left 和 top 如何区分。
- 过滤掉多少样本、为什么过滤、每类过滤比例是多少，没有充分报告。
- 是否有多标注者协议、inter-annotator agreement、冲突解决流程，没有足够透明。
- 合成子任务是否代表真实 OCR error，生态有效性不够清楚。

这类问题对 benchmark 很致命，因为 benchmark 的价值来自社区信任。如果读者不知道题目怎么来的、错误怎么排除、标注是否一致，就很难相信它能作为标准评测。

### 2.2 OCRAssistor 依赖外部 OCR，鲁棒性分析不足

OCRAssistor 的核心是用外部轻量 OCR 模型给 LVLM 解码提供约束。但外部 OCR 也会错。审稿人担心：

- 如果 OCR 漏检或误识别，方法可能放大错误。
- 方法是否会把 faulty OCR cue 变成新的 hallucination。
- 论文中对 severe OCR failure 的实验和错误分析不够深入。

作者 rebuttal 后补了 correct/incorrect OCR 子集实验，但 AC 仍认为该问题没有完全解决。

### 2.3 实验比较不够充分

Reviewer ne8t 认为比较对象太少，导致无法判断 OCRAssistor 的真实优势。缺少的比较包括：

- calibration 方法；
- consensus 或 ensemble 方法；
- 其他 contrastive decoding 变体；
- 更强 OCR 或 document parsing 系统，例如 MinerU、MinerU2.5；
- pipeline-based OCR 系统；
- document parsing benchmark，例如 OmniDocBench。

问题本质是：提升到底来自 OCRAssistor 的机制，还是来自“把 OCR 信息接进来了”？如果 baseline 不够强，方法贡献就站不稳。

### 2.4 泛化范围偏窄

多个 reviewer 提到，HalluText 主要是 multiple-choice OCR/recognition 设置，而真实场景往往需要：

- open-ended transcription；
- document captioning；
- DocQA / ChartQA 这类视觉 + OCR + reasoning；
- long-context document understanding；
- document parsing。

如果论文声称方法通用，但评测主要集中在选择题式 benchmark 上，审稿人会认为 scope 太窄。

### 2.5 方法设计和 presentation 不够成熟

审稿人还关注：

- 为什么选择 KL-divergence guidance，而不是其他 distribution modification 方法。
- 温度 T 和 regularization weight lambda 是否需要针对任务调参。
- prompt ablation 是否公平。
- OCRAssistor 在哪些子任务上有帮助，在哪些子任务上失败。
- 写作中是否有 overclaim，例如把 benchmark 设计说成 theoretical grounding。
- 论文格式和表达是否一致。

这说明 benchmark paper 不只要有数据，还要有足够强的叙事、定义、实验与写作纪律。

## 3. 做 Benchmark 时最重要的可信度链条

一个 benchmark 要被 reviewer 接受，通常需要回答下面这条链：

1. 我到底在测什么能力或失败模式？
2. 这个能力是否被清楚定义，并和已有 benchmark 区分开？
3. 每一道题是否真的测这个能力？
4. 答案是否唯一且可验证？
5. distractor 是否公平，不靠投机或歧义制造难度？
6. 标注是否稳定，不依赖单个作者主观判断？
7. 任务是否有足够的模型区分度？
8. baseline 是否覆盖了简单但强的替代方案？
9. 分项分析是否能解释模型为什么错？
10. 数据是否可复现、可维护、可扩展？

可以把它记成一句话：定义要窄，构建要透明，标注要一致，评测要公平，分析要能诊断。

## 4. Benchmark 设计的具体建议

### 4.1 先把“测什么”定义窄

不要一开始写：

- 我们评估 OCR hallucination。
- 我们评估 multimodal reasoning。
- 我们评估 robustness。

这些都太大。更好的写法是把能力定义成可判定的 construct，例如：

- 模型是否能抵抗语言先验，读出图中非常规拼写？
- 模型是否能定位指定文本区域，而不是凭常识猜？
- 模型是否能在 OCR 正确但布局复杂时完成 reasoning？
- 模型是否会在图中不存在文本时编造答案？
- 模型是否能区分视觉证据和世界知识冲突？

每个 construct 都应该配：

- 定义；
- 正例；
- 反例；
- 边界情况；
- 为什么已有 benchmark 测不到；
- 对应的自动或人工判定规则。

### 4.2 Taxonomy 要能落到题目生成规则

好的 taxonomy 不是概念列表，而是数据构建说明书。每一类都要能回答：

- 该类任务的输入是什么？
- 正确答案如何产生？
- 错误选项如何产生？
- 哪些样本必须删除？
- 该类任务最容易产生什么歧义？
- 用什么规则消除歧义？

例如 Position 类任务不能只说“判断文本相对位置”，还需要说明：

- 用 bounding box 中心点还是 polygon center；
- 如何计算相对角度；
- 角度区间如何划分；
- 边界附近是否丢弃；
- 文本之间距离太近如何处理；
- 当正确答案是 top-left 时，top 和 left 是否可以作为 distractor。

### 4.3 数据构建流程要像实验 protocol 一样写

建议在论文中明确写出 pipeline：

1. Data sources：来自哪些公开数据集、真实场景或合成流程。
2. Candidate extraction：如何抽取图像、文本区域、metadata。
3. Filtering：清晰度、遮挡、语言、重复文本、低质量 OCR、异常尺寸等规则。
4. Task generation：人工、模板、规则、LLM 生成分别怎么用。
5. Answer derivation：答案来自人工标注、OCR box、原始 metadata 还是程序计算。
6. Distractor generation：每类 distractor 的生成规则。
7. Human verification：几名标注者检查哪些内容。
8. Conflict resolution：不一致时如何处理。
9. Final statistics：每一步保留和删除多少样本。

不要只写“we carefully filter”或“we manually verify”。要写具体数字和协议。

### 4.4 Distractor 是 multiple-choice benchmark 的核心

如果 benchmark 是选择题，distractor 的设计决定它是在测能力，还是在测投机。

OCR / 视觉文本任务中可以设计多层 distractor：

| Distractor 类型 | 示例 | 目的 |
| --- | --- | --- |
| 字符级 | O vs 0，rn vs m，l vs I | 测细粒度识别 |
| 拼写级 | COFFEE vs KOFFEE | 测语言先验干扰 |
| 位置级 | 相邻区域文本 | 测 grounding |
| 数值级 | 计数 +1 / -1 | 测 counting |
| 顺序级 | 交换字符或词序 | 测局部顺序 |
| 常识诱导 | 图里写反常内容 | 测视觉证据是否压过常识 |

好的 distractor 应满足两点：

- Plausible：看起来合理，不能太弱。
- Falsifiable：视觉证据能唯一排除，不能有歧义。

建议报告 distractor 的质量控制：

- 每类 distractor 的生成规则；
- 是否排除 near-duplicate；
- 是否排除语义上也可能正确的选项；
- 是否进行人工验证；
- 是否统计选项位置偏差；
- 是否避免答案分布不均衡。

### 4.5 合成数据要说明它的角色

合成数据不是问题，但要讲清楚它不是自然分布的替代品，而是 controlled diagnostic probe。

例如 Stroop、Typo、Incompletion 任务可以这样定位：

- 它们不是为了模拟真实世界 OCR 错误的完整分布；
- 它们用于隔离模型对视觉文本和语言先验冲突的反应；
- 它们是 diagnostic subset，而不是 deployment distribution。

如果有合成数据，最好补充 sanity check：

- 合成 typo 是否覆盖真实 OCR confusion pattern；
- 字体、背景、旋转、模糊、遮挡是否多样；
- 在真实数据中是否也观察到同类 failure；
- 合成子集上的模型排序是否和真实子集相关；
- 人类是否能稳定识别这些样本。

### 4.6 标注协议要可审计

Benchmark paper 最好给出 annotation protocol：

- 标注员数量；
- 标注员背景；
- 是否独立标注；
- 是否 blind to model outputs；
- 每个样本标注什么字段；
- disagreement 如何解决；
- 是否计算 Cohen's kappa、Krippendorff's alpha 或 raw agreement；
- 删除样本的原因统计。

一个强写法：

> Two annotators independently verified each candidate. A sample was retained only if both annotators agreed that the visual evidence was sufficient, the answer was unique, and all distractors were falsifiable. Disagreements were adjudicated by a third annotator. We report agreement for task labels, answer validity, and ambiguity flags.

### 4.7 要证明 benchmark 有区分度

Benchmark 不是越难越好，而是要能产生有意义的模型排序和诊断。

至少需要：

- Random baseline；
- Majority / heuristic baseline；
- Text-only baseline；
- Image-only baseline；
- OCR-only baseline；
- OCR + LLM prompt baseline；
- strong public OCR pipeline；
- 主流开源 LVLM；
- 强闭源模型；
- human performance。

还要报告：

- overall score；
- per-category score；
- easy / medium / hard 分层；
- 不同数据源分层；
- 不同语言或字体分层；
- 模型规模趋势；
- 是否存在 shortcut。

如果强模型、人类、OCR-only、random 的相对位置都合理，benchmark 的可信度会高很多。

### 4.8 错误分析要解释“为什么错”

只报 accuracy 不够。需要系统分析：

- OCR 错了，导致模型被误导；
- OCR 对了，但 LVLM 忽略了 OCR cue；
- LVLM 视觉定位错；
- LVLM 读对了文本，但 reasoning 错；
- 题目本身存在歧义；
- distractor 太接近；
- prompt 格式导致误解。

最好给出 confusion matrix 或 error taxonomy，并说明每类错误占比。这样 reviewer 才会觉得 benchmark 能产生 insight，而不只是排名榜。

## 5. 可直接套用的 Benchmark 论文结构

### 5.1 Problem Definition

写清楚：

- 目标能力是什么；
- 失败模式是什么；
- 与已有任务的区别；
- 为什么这个问题重要；
- 为什么现有 benchmark 不够。

### 5.2 Taxonomy

每个子类包含：

- definition；
- positive examples；
- negative examples；
- boundary cases；
- construction rule；
- exclusion rule。

### 5.3 Dataset Construction

建议结构：

1. Source datasets；
2. Candidate collection；
3. Task-specific generation；
4. Distractor generation；
5. Filtering and quality control；
6. Human annotation；
7. Dataset statistics。

### 5.4 Evaluation Protocol

需要明确：

- prompt 模板；
- decoding 参数；
- 是否允许外部 OCR；
- 是否允许 CoT；
- open-source 和 closed-source 模型如何统一评测；
- 指标；
- 多次采样如何处理；
- 答案解析规则。

### 5.5 Baselines

建议从弱到强：

- Random；
- heuristic；
- OCR-only；
- LLM-only；
- LVLM baseline；
- LVLM + OCR prompt；
- LVLM + CoT；
- SOTA public systems；
- human。

### 5.6 Validation

至少包括：

- human agreement；
- human upper bound；
- model separability；
- subset difficulty；
- shortcut analysis；
- synthetic-real correlation；
- ablation of distractor difficulty；
- data leakage check。

### 5.7 Limitations

主动写：

- 该 benchmark 不覆盖哪些场景；
- 合成数据不能代表哪些真实分布；
- 语言、字体、领域、图像来源的局限；
- 模型分数不能外推到哪些应用；
- 后续版本计划。

## 6. 如果我自己做 Benchmark，可以这样落地

第一版不要追求大而全，建议做一个“窄但铁”的 benchmark：

- 子任务控制在 3 到 5 类；
- 每类任务定义非常清楚；
- 每类至少有可复现的生成规则；
- 每个样本都有质量验证字段；
- 每个 distractor 都有生成原因；
- 做强 baseline，而不是只和弱模型比；
- 提供 data card；
- 提供 annotation guideline；
- 提供 public dev set 和 held-out test set。

一个最低可发表版本可以包含：

- 1,000 到 3,000 个高质量样本；
- 3 到 5 个明确 failure modes；
- 2 名标注者 + 1 名仲裁；
- 至少 8 到 12 个模型；
- random / OCR-only / OCR+LLM / human baselines；
- 每类任务的 error analysis；
- 数据构建脚本或伪代码；
- data card 和 examples。

## 7. 从 HalluText 学到的避坑清单

提交前逐项检查：

- 是否把每个子任务的边界情况写清楚？
- 是否报告每一步过滤掉多少样本？
- 是否说明 QA pair 是模板、规则、人工还是 LLM 生成？
- 是否说明 distractor 如何生成、如何去歧义？
- 是否给出标注协议和 agreement？
- 是否有人类表现作为上限？
- 是否有 OCR-only 和 OCR+LLM 这种简单强 baseline？
- 是否和已有 SOTA OCR/document parsing 系统比较？
- 是否有 open-ended 或真实任务泛化实验？
- 是否分析方法失败案例？
- 是否解释为什么选择某个方法设计，而不是其他替代设计？
- 是否避免 overclaim？
- 是否把 rebuttal 里才说清楚的内容提前写进主文？

## 8. 参考论文与资料

以下论文适合作为 benchmark 设计参考：

- Datasheets for Datasets  
  https://arxiv.org/abs/1803.09010  
  学习如何系统记录数据集的动机、组成、收集、标注、用途和限制。

- Data Cards  
  https://arxiv.org/abs/2204.01075  
  学习如何把数据集文档做成结构化、可审计的说明。

- HELM: Holistic Evaluation of Language Models  
  https://arxiv.org/abs/2211.09110  
  学习如何先定义 scenarios 和 metrics，再做多维评测。

- TextVQA  
  https://arxiv.org/abs/1904.08920  
  学习如何把“读图中文字并推理”定义成新 benchmark。

- ST-VQA  
  https://arxiv.org/abs/1905.13648  
  学习 scene text VQA 的任务设计和评测协议。

- OCRBench  
  https://arxiv.org/abs/2305.07895  
  学习 OCR-centric multimodal benchmark 的任务覆盖和模型比较。

- OCRBench v2  
  https://arxiv.org/abs/2501.00321  
  学习更大规模 OCR 评测如何组织任务、场景和 human verification。

- ChartQA  
  https://arxiv.org/abs/2203.10244  
  学习如何结合模板题和人工题，评估图表中的视觉与逻辑推理。

- DocVQA  
  https://arxiv.org/abs/2007.00398  
  学习 document VQA benchmark 的设计方式。

- OmniDocBench  
  https://arxiv.org/abs/2412.07626  
  学习文档解析 benchmark 的细粒度标注、版面结构和多层级指标。

- M3CoT  
  https://arxiv.org/abs/2405.16473  
  学习如何论证已有 benchmark 缺少 multi-domain、multi-step、multi-modal reasoning。

## 9. 可以直接复制进论文的 Method 小纲

```text
We construct [Benchmark Name] to evaluate [specific capability/failure mode].
Unlike prior benchmarks that focus on [existing scope], our benchmark targets
[precise gap] through [number] diagnostic subsets.

For each subset, we define a task-specific construction rule, answer derivation
procedure, distractor generation strategy, and ambiguity filtering criterion.
Candidate samples are first collected from [sources], then filtered according to
[visual quality / text uniqueness / layout / language / ambiguity] constraints.

Each candidate is independently verified by [N] annotators. A sample is retained
only if annotators agree that the visual evidence is sufficient, the answer is
unique, and all distractors are falsifiable. Disagreements are resolved by
[protocol]. We report inter-annotator agreement for [labels].

We evaluate models under [closed-book / OCR-allowed / OCR-free] settings using
the same prompt template and decoding configuration. We include random,
heuristic, OCR-only, LVLM-only, OCR+LLM, and human baselines to measure both
task difficulty and model separability.
```

## 10. 下一步建议

如果后续要正式设计自己的 benchmark，第一步不是收数据，而是写一页 spec：

- 任务名；
- 要测的能力；
- 不测的能力；
- 3 到 5 个子任务；
- 每个子任务的正例、反例、边界情况；
- 数据来源；
- 标注字段；
- baseline 列表；
- 预期最强模型会在哪些地方失败。

这页 spec 写清楚后，再开始收数据，会少走很多弯路。
