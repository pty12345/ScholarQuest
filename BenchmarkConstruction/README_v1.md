# PaperSearch Benchmark Query Taxonomy Specification

## 1. 文档目标

本文档定义 PaperSearch Benchmark 的 query taxonomy，用于指导 benchmark query 的生成、标注、采样与质量控制。

该 benchmark 的设计目标不是覆盖所有理论上可能的论文检索需求，而是聚焦于：

- 当前主流 paper search 方法可处理的输入范围
- 主要依赖论文标题与摘要即可完成判断的检索需求
- 同时能够超越简单关键词匹配或纯 embedding 相似度匹配的 query 类型

因此，本 benchmark **不以细粒度实验设置、数据集使用细节、正文方法细节抽取** 为主要目标，而是强调：

- 具体主题理解
- 多概念组合检索
- 方法/架构属性识别
- 任务能力映射
- 论文集合构建
- 粗粒度分析性检索

------

## 2. 总体设计原则

### 2.1 Query 纳入标准

一个 query 适合进入主 benchmark，当它满足以下条件：

1. query 的主要判定依据通常能在论文标题或摘要中找到
2. query 不应退化为单纯的宽泛主题词匹配
3. query 需要一定程度的语义理解、概念组合、属性识别或结果集合组织能力
4. query 的边界相对可判定，适合构建可复核的 ground truth

### 2.2 Query 排除标准

以下类型不建议纳入主 benchmark：

1. 明确依赖正文实验部分的信息
   例如：
   - “在 HotPotQA 上评测”
   - “在 zero-shot setting 下优于某基线”
   - “使用某个 appendix 中定义的训练 trick”
2. 细粒度训练或实现细节
   例如：
   - 某个 loss 变体
   - 某个超参数设置
   - 仅在表格或附录中出现的实现差异
3. 强依赖结果表比较的 query
   例如：
   - “优于方法 A 在数据集 B 上的结果”
   - “比 supervised small model 更强”

这些类型可以作为 future extension 或 challenge split，但不应构成主 benchmark 的主体。

------

## 3. 一级类型定义

主 benchmark 包含以下 6 类 query：

1. Specific Topic Search
2. Cross-topic Compositional Search
3. Method/Architecture-based Search
4. Capability- or Application-oriented Search
5. Collection-building and Literature Scoping Search
6. Coarse-grained Analysis or Claim-oriented Search

------

# 4. Type 1: Specific Topic Search

## 4.1 定义

Specific Topic Search 指针对某个**相对具体、边界较清晰的研究主题**进行论文检索的 query。

这类 query 通常聚焦于一个明确研究方向，而非过于宽泛的大领域词。

## 4.2 设计动机

这类 query 是 benchmark 的基础组成部分，用于测试系统是否能完成：

- 具体主题语义理解
- 学术表达归一化
- 相近主题之间的区分

虽然这类 query 相对基础，但若主题足够具体，仍然能够区分简单关键词检索与较强语义检索系统。

## 4.3 纳入标准

适合纳入的 query 应满足：

- 主题具体，不宜过宽
- 主题通常可由标题或摘要直接表达
- 不依赖正文实验细节

## 4.4 不宜纳入的情况

以下情况不应归入此类：

- 主题过宽，如 “papers on LLM”
- 实际上包含额外隐式约束，应归入其他类
- 依赖正文细节判断的主题

## 4.5 示例

- papers on long video description
- research on trigger-free document-level event extraction
- work on identity-preserving video generation
- papers about LLM quantized pretraining

## 4.6 生成建议

生成这类 query 时：

- 优先选择中等粒度主题
- 避免过泛化的大领域名词
- 尽量使用学术上稳定存在的主题表述

## 4.7 推荐标注字段

- `type`: `specific_topic`
- `topic_span`: 主题短语
- `scope`: `narrow` or `medium`
- `difficulty`: `easy` to `medium`

------

# 5. Type 2: Cross-topic Compositional Search

## 5.1 定义

Cross-topic Compositional Search 指 query 同时包含两个或多个研究概念，目标是检索它们交叉区域中的论文。

这类 query 的重点不是单一主题，而是多个概念之间的组合关系。

## 5.2 设计动机

这类 query 是主 benchmark 的核心组成之一，因为它明显超越简单单主题匹配。系统需要同时理解多个概念，并保证返回结果真正落在它们的交集上，而不是只匹配其中一部分。

## 5.3 纳入标准

适合纳入的 query 应满足：

- 至少包含两个重要概念
- 组合后的意图通常能从标题或摘要判断
- 组合关系较自然，符合真实学术检索习惯

## 5.4 不宜纳入的情况

以下情况不应归入此类：

- 组合概念过多，导致语义极端稀疏
- 其中某个关键约束必须依赖正文才能判断
- 实际是方法属性或应用需求，应归入其他类

## 5.5 示例

- reinforcement learning for LLM agents
- multimodal foundation models with visual and audio inputs
- diffusion models for video generation
- LLMs for stock factor mining
- robot decision making and task planning

## 5.6 生成建议

生成这类 query 时：

- 优先使用 2 个核心概念组合
- 必要时可扩展到 3 个，但不宜过多
- 组合中的每个概念都应在学术文献中有稳定表述
- 避免“主题 + 实验细节”式组合

## 5.7 推荐标注字段

- `type`: `cross_topic`
- `concepts`: 概念列表
- `num_concepts`: 2 or 3
- `compositionality`: `medium` or `high`
- `difficulty`: `medium` to `hard`

------

# 6. Type 3: Method/Architecture-based Search

## 6.1 定义

Method/Architecture-based Search 指 query 以某种**方法、训练范式、模型结构或架构属性**为核心检索条件。

重点在于识别论文“采用了什么方法”或“属于什么架构范式”。

## 6.2 设计动机

这类 query 测试系统是否能够从标题和摘要中识别：

- 方法类别
- 训练范式
- 架构家族
- 显式方法属性

它比纯主题检索更细一层，但仍然通常不需要阅读正文。

## 6.3 纳入标准

适合纳入的 query 应满足：

- 方法或架构属性通常会在标题或摘要中明确提及
- query 关注的是粗粒度方法属性，而非实现细节
- 返回结果可通过标题/摘要较稳定判断

## 6.4 不宜纳入的情况

以下情况不应归入此类：

- 只在方法章节或附录中出现的细节
- 极细粒度训练 trick
- 必须读实验表才能确认的方法差异

## 6.5 示例

- autoregressive transformer for video generation
- visual LLMs with MoE architecture
- DPO for vision-language models
- RLHF for hallucination reduction
- quantization-aware training for low-bit models
- rejection sampling finetuning for LLMs

## 6.6 生成建议

生成这类 query 时：

- 优先选取社区常用术语，如 DPO、RLHF、MoE、autoregressive transformer
- 方法属性应为标题或摘要常出现的信息
- 方法条件可与任务结合，但不要叠加正文级约束

## 6.7 推荐标注字段

- `type`: `method_architecture`
- `method_family`: 方法族
- `architecture_family`: 架构族
- `training_paradigm`: 训练范式
- `difficulty`: `medium`

------

# 7. Type 4: Capability- or Application-oriented Search

## 7.1 定义

Capability- or Application-oriented Search 指 query 从模型能力、任务功能或应用需求出发，而不是直接使用标准论文标题式术语。

这类 query 往往描述“模型能做什么”或“系统被用于什么场景”。

## 7.2 设计动机

真实用户在检索论文时，经常不是按标准术语发问，而是按需求表达目标。
因此这类 query 可以测试系统是否能够将自然需求映射到学术研究问题。

## 7.3 纳入标准

适合纳入的 query 应满足：

- query 描述某种能力或应用目标
- 相关论文的标题或摘要通常会描述该能力或应用
- 不依赖实验设置或细节指标来判断

## 7.4 不宜纳入的情况

以下情况不应归入此类：

- 需求过于开放，边界不清
- 应用描述缺乏稳定学术对应术语
- 实际上是在做细粒度结论验证

## 7.5 示例

- LLM agents can do schedule planning
- models that automatically write surveys from multiple papers
- VLM agents that can play PC games
- work on controllable video generation
- models for long video understanding
- LLMs for mining factors in stock exchange analysis

## 7.6 生成建议

生成这类 query 时：

- 可以使用相对自然的用户表达
- 但仍需保证存在较稳定的学术对应方向
- 优先选择能力边界明确的任务
- 避免特别口语化、极难规范化的表述

## 7.7 推荐标注字段

- `type`: `capability_application`
- `capability`: 能力描述
- `application_domain`: 应用领域
- `terminology_explicitness`: `explicit` or `implicit`
- `difficulty`: `medium` to `hard`

------

# 8. Type 5: Collection-building and Literature Scoping Search

## 8.1 定义

Collection-building and Literature Scoping Search 指 query 的目标不是找到单篇最相关论文，而是构建某一方向上的**论文集合、阅读列表或研究范围**。

这类 query 强调结果集合的边界与覆盖性。

## 8.2 设计动机

真实 paper search 使用场景中，用户经常希望：

- 搭 related work 列表
- 收集某方向代表性论文
- 梳理某一子方向的研究边界

因此这类 query 用于测试系统的：

- set-level recall
- 去重能力
- 范围控制能力
- literature scoping 能力

## 8.3 纳入标准

适合纳入的 query 应满足：

- 检索目标是一个论文集合，而非单篇答案
- 集合边界主要可以从标题和摘要判定
- query 所描述的研究范围相对清楚

## 8.4 不宜纳入的情况

以下情况不应归入此类：

- 范围极度开放，几乎无边界
- 需要依赖正文做细粒度筛选
- 实际上是强约束筛选型 query

## 8.5 示例

- all papers on machine translation agents
- research on controllable video generation
- papers on long video description
- visual-LLM models with MoE architecture
- datasets and benchmarks for robot task planning

## 8.6 生成建议

生成这类 query 时：

- 可使用 “all papers on”, “research on”, “works on”, “datasets and benchmarks for” 等模板
- 主题边界需要相对稳定
- 不宜使用过于模糊的时间、影响力或主观评价词

## 8.7 推荐标注字段

- `type`: `collection_scoping`
- `collection_goal`: `all_papers` / `representative_works` / `datasets_benchmarks`
- `scope`: `narrow` / `medium`
- `recall_requirement`: `high`
- `difficulty`: `medium`

------

# 9. Type 6: Coarse-grained Analysis or Claim-oriented Search

## 9.1 定义

Coarse-grained Analysis or Claim-oriented Search 指 query 关注某种**较粗粒度的现象、分析主题、结论方向或机制解释**，且这些内容通常会在标题或摘要中直接表达。

这类 query 不要求系统判断细粒度实验结果，只要求定位研究某种现象或明确表达某一分析方向的论文。

## 9.2 设计动机

这类 query 可以测试系统是否能检索：

- phenomenon analysis
- mechanism explanation
- broad conclusion-oriented work

它比一般 topic search 更偏分析导向，但又不至于依赖结果表和实验细节。

## 9.3 纳入标准

适合纳入的 query 应满足：

- claim 或 analysis 较粗粒度
- 标题或摘要中通常会显式表达该分析主题
- 不要求依赖正文做精确证伪或证实

## 9.4 不宜纳入的情况

以下情况不应归入此类：

- 需要读实验表才能确认的精确结论
- 涉及严格的 “优于/劣于” 数值判断
- 结论边界非常主观或歧义极大

## 9.5 示例

- research on how in-context learning emerges during pretraining
- papers on scaling laws for multimodal models
- work discussing self-correction failure in LLMs
- research on image encoding distributions
- papers analyzing long-context capability in LLMs

## 9.6 生成建议

生成这类 query 时：

- 优先使用 how / why / analysis / scaling laws / failure / emergence / mechanism 等表达
- 避免细粒度比较句式
- 保持 claim 为粗粒度、可被标题摘要直接表达的现象级内容

## 9.7 推荐标注字段

- `type`: `analysis_claim`
- `analysis_target`: 分析对象
- `claim_polarity`: `neutral` / `positive` / `negative`
- `granularity`: `coarse`
- `difficulty`: `hard`

------

# 10. 类型间区分规则

为了方便后续写代码，建议加入一个简单的优先级判断规则。

## 10.1 优先级建议

当一个 query 同时符合多类时，按以下优先级分配主类型：

1. collection_scoping
2. analysis_claim
3. method_architecture
4. capability_application
5. cross_topic
6. specific_topic

## 10.2 判定规则示例

### Rule A

如果 query 明确要求“all papers / research on / representative works / datasets and benchmarks”，优先归为 `collection_scoping`。

### Rule B

如果 query 关注某种“现象、机制、failure、scaling law、emergence”，优先归为 `analysis_claim`。

### Rule C

如果 query 的中心约束是某种方法、训练范式或架构属性，优先归为 `method_architecture`。

### Rule D

如果 query 明显从能力需求或应用场景出发，而不是论文术语出发，优先归为 `capability_application`。

### Rule E

如果 query 明显是两个或多个概念的交叉组合，归为 `cross_topic`。

### Rule F

若以上都不满足，但存在一个较具体主题，则归为 `specific_topic`。

------

# 11. 建议的数据结构

下面是推荐的统一 schema。

```json
{
  "query": "string",
  "canonical_query": "string",
  "type": "specific_topic | cross_topic | method_architecture | capability_application | collection_scoping | analysis_claim",
  "topics": ["string"],
  "methods": ["string"],
  "applications": ["string"],
  "analysis_targets": ["string"],
  "scope": "narrow | medium | broad",
  "difficulty": "easy | medium | hard",
  "terminology_explicitness": "explicit | implicit",
  "recall_requirement": "low | medium | high",
  "notes": "string"
}
```

------

# 12. Query 生成约束

后续用代码生成 query 时，建议遵循以下规则：

## 12.1 长度约束

- query 尽量保持自然
- 不宜过长
- 一般控制在 4 到 20 个词之间较合适

## 12.2 表达约束

- 避免包含过多细粒度实验条件
- 避免使用需要全文阅读才能判断的约束
- 避免时间敏感或主观性太强的表达，如 “latest”, “best”, “most influential”

## 12.3 术语约束

- 尽量使用社区中稳定存在的术语
- 若使用偏自然语言表达，也应能映射到较稳定的学术任务

## 12.4 难度控制

- easy：单一具体主题
- medium：两概念组合，或显式方法属性
- hard：应用需求映射、分析导向、边界较复杂的集合构建

------

# 13. 建议的类型分布

主 benchmark 可参考如下分布：

- specific_topic: 20%
- cross_topic: 25%
- method_architecture: 20%
- capability_application: 15%
- collection_scoping: 15%
- analysis_claim: 5%

这个分布的考虑是：

- 保证基础主题检索存在
- 强化 compositional search 和 method-aware search
- 让 benchmark 不被 claim 型歧义 query 主导

------

# 14. 质量检查规则

生成 query 后，建议对每条 query 做以下检查：

## 14.1 可判定性检查

query 是否主要依赖标题和摘要即可判断？

## 14.2 边界清晰性检查

query 的检索边界是否相对明确？

## 14.3 非退化检查

query 是否不是过宽泛的主题词？

## 14.4 非正文依赖检查

query 是否不依赖实验表、附录或方法细节？

## 14.5 类型一致性检查

该 query 是否与标注的 taxonomy type 一致？

------

# 15. 简化版总结

PaperSearch Benchmark 主 benchmark 应聚焦以下六类 query：

1. **Specific Topic Search**
   针对具体研究主题进行检索。
2. **Cross-topic Compositional Search**
   检索两个或多个概念交叉区域的论文。
3. **Method/Architecture-based Search**
   检索采用特定方法、训练范式或架构属性的论文。
4. **Capability- or Application-oriented Search**
   从能力需求或应用场景出发检索相关论文。
5. **Collection-building and Literature Scoping Search**
   构建某个方向的论文集合、阅读列表或研究范围。
6. **Coarse-grained Analysis or Claim-oriented Search**
   检索研究某种现象、机制、分析主题或粗粒度结论方向的论文。