# Benchmark Query Constraint Report

## Scope

- Source files: `PASA_Benchmark.jsonl`, `SPAR_Benchmark.jsonl`
- Total extracted queries: `100`
- Focus: classify surface-level query constraints rather than research topics
- Method: rule-based multi-label tagging with manual taxonomy design

## High-Level Findings

- The benchmark queries usually mix topic intent with retrieval constraints, evidence requirements, and answer-format instructions.
- `PASA_Benchmark.jsonl` is closer to imperative search requests, while `SPAR_Benchmark.jsonl` is closer to open-form research questions.
- Constraint composition is uneven: a small number of constraints appear often, while many queries remain mostly topic-driven and lightly constrained.
- For future benchmark design, it is useful to separate topic diversity from constraint diversity and control both explicitly.

## Source-Level Style Comparison

### `PASA_Benchmark.jsonl`

- Query count: `50`
- Imperative retrieval style: `28`
- Question form: `7`
- Contains question mark: `6`
- Multi-sentence or multi-clause queries: `5`
- Long-form queries: `5`

### `SPAR_Benchmark.jsonl`

- Query count: `50`
- Imperative retrieval style: `11`
- Question form: `31`
- Contains question mark: `37`
- Multi-sentence or multi-clause queries: `21`
- Long-form queries: `21`

## Constraint Taxonomy

### Document Type Constraint

- Definition: The query restricts the requested output to a specific literature or artifact type.
- Overall count: `31`
- `PASA_Benchmark.jsonl`: `10`
- `SPAR_Benchmark.jsonl`: `21`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_0`: Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_3`: I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_6`: Papers that propose methods based on large language models and evaluate their performance through experiments on the HotPotQA dataset.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_14`: Find papers that use LLMs or LLM-based agents to automatically write surveys or summaries for multiple scholarly documents.

### Include / Exclude Constraint

- Definition: The query explicitly includes or excludes a subclass of results.
- Overall count: `4`
- `PASA_Benchmark.jsonl`: `2`
- `SPAR_Benchmark.jsonl`: `2`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_3`: I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_16`: Find papers on trigger-free document-level event extraction methods that do not use human-annotated triggers.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:20`: How can the performance of multi-task learning models be improved, especially in complex applications? For example, in developing a translation software, I want it not only to translate languages but also to correct grammar and enhance fluency and readability.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:21`: How can self-supervised learning improve natural language processing tasks without a large amount of labeled data? If we only have some unlabeled data, can performance still be improved?

### Exhaustive Scope Constraint

- Definition: The query asks for a complete or near-complete collection rather than a few examples.
- Overall count: `5`
- `PASA_Benchmark.jsonl`: `5`
- `SPAR_Benchmark.jsonl`: `0`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_2`: List all papers that use autoregressive transformer to generate videos.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_4`: Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_11`: Give me all visual-LLM models that are MoE architecture
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_24`: Show me all research papers on machine translation agents.

### Recency / Venue Quality Constraint

- Definition: The query prefers recent, cutting-edge, popular, or top-tier work.
- Overall count: `14`
- `PASA_Benchmark.jsonl`: `1`
- `SPAR_Benchmark.jsonl`: `13`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_33`: Show me some popular papers on generating textual adversarial examples for machine translation.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:1`: Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:2`: Show some cutting-edge technological advancements on how to improve the generalization ability of machine learning models across multiple domains.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:5`: How can machine learning be applied to climate prediction, especially in data-scarce scenarios? What are the cutting-edge methods to improve prediction accuracy? Please explain in detai.

### Comparison / Baseline Constraint

- Definition: The query is framed around outperforming, underperforming, or being between known baselines.
- Overall count: `5`
- `PASA_Benchmark.jsonl`: `5`
- `SPAR_Benchmark.jsonl`: `0`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_13`: Provide papers demonstrating that the self-correction of LLMs does not enhance their performance.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_15`: Provide papers claiming that reinforcement learning can negatively impact the performance of supervised fine-tuned LLMs.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_17`: Provide papers explaining why the in-context learning performance of LLMs cannot surpass that of supervised fine-tuned small language models in information extraction tasks, such as NER, RE, and EE.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_18`: Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.

### Threshold / Definition Constraint

- Definition: The query gives an explicit threshold, operational definition, or scale condition.
- Overall count: `16`
- `PASA_Benchmark.jsonl`: `6`
- `SPAR_Benchmark.jsonl`: `10`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_3`: I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_7`: Show me research on the long video description. Here, long videos are defined as those with a duration of at least several minutes.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_18`: Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_28`: Show me code evaluation datasets with a mid-level hardness. It show be harder than HumanEval and MBPP, but easier than code_contests.

### Method / Capability Constraint

- Definition: The query requires a method, architecture, capability, modality, or training setup to be present.
- Overall count: `14`
- `PASA_Benchmark.jsonl`: `12`
- `SPAR_Benchmark.jsonl`: `2`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_0`: Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_2`: List all papers that use autoregressive transformer to generate videos.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_3`: I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_4`: Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.

### Evaluation / Setting Constraint

- Definition: The query anchors the search to a dataset, benchmark, application setting, or experimental environment.
- Overall count: `5`
- `PASA_Benchmark.jsonl`: `3`
- `SPAR_Benchmark.jsonl`: `2`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_4`: Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_6`: Papers that propose methods based on large language models and evaluate their performance through experiments on the HotPotQA dataset.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_33`: Show me some popular papers on generating textual adversarial examples for machine translation.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:4`: Provide me with some research papers evaluating the application performance of large-scale language models (LLMs) in the financial sector.

### Output Style Constraint

- Definition: The query asks for explanation, analysis, or a specific answer presentation style in addition to paper retrieval.
- Overall count: `4`
- `PASA_Benchmark.jsonl`: `2`
- `SPAR_Benchmark.jsonl`: `2`
- Representative queries:
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_1`: Give me papers that share some insights about how large language models gain in-context learning capability in the process of pre-training.
  - `PASA_Benchmark.jsonl` / `RealScholarQuery_18`: Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:3`: How can deep learning enhance the perception and decision-making accuracy of autonomous driving systems? Please provide a comprehensive analysis with supporting research papers.
  - `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:5`: How can machine learning be applied to climate prediction, especially in data-scarce scenarios? What are the cutting-edge methods to improve prediction accuracy? Please explain in detai.

## Distribution Notes

- `Document Type Constraint` appears in `31` queries.
- `Threshold / Definition Constraint` appears in `16` queries.
- `Method / Capability Constraint` appears in `14` queries.
- `Recency / Venue Quality Constraint` appears in `14` queries.
- `Exhaustive Scope Constraint` appears in `5` queries.
- `Evaluation / Setting Constraint` appears in `5` queries.
- `Comparison / Baseline Constraint` appears in `5` queries.
- `Output Style Constraint` appears in `4` queries.
- `Include / Exclude Constraint` appears in `4` queries.
- Queries without any current rule match: `37`

## Design Implications For New Benchmark Queries

- Separate the topic slot from the constraint slot. A future query template should let you compose them independently.
- Add explicit contrastive buckets such as `include only`, `exclude`, `between baselines`, and `threshold-defined scope` because they are frequent but not evenly covered.
- Keep both imperative search requests and research-question style prompts. They stress retrieval systems differently.
- Avoid mixing too many constraints in a single query unless that is an intentional hard setting. Some existing queries combine topic, modality, scale, exclusion, and answer-style requirements at once.
- Track whether a constraint is retrieval-facing or answer-format-facing. These are different difficulty sources.
- Consider building a balanced benchmark matrix: `topic family x constraint family x surface form`.

## Suggested Constraint Families For Future Query Authoring

- `Document type`: survey, dataset, benchmark, journal paper, workshop paper
- `Include / exclude`: only, except, without, exclude
- `Scope`: all papers, representative papers, a few examples, exhaustive list
- `Time / quality`: latest, classic, top-tier, popular, influential
- `Comparison`: better than, worse than, harder than, between A and B
- `Definition / threshold`: at least N, long-form defined as X, large-scale defined as Y
- `Method / capability`: must use RLHF, must support audio, must be autoregressive
- `Evaluation / setting`: on HotPotQA, in finance, for robotics, under data scarcity
- `Output style`: explain why, provide analysis, summarize evidence

## Full Extracted Query Inventory

### `PASA_Benchmark.jsonl`

- `RealScholarQuery_0`
  - Query: Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.
  - Constraint labels: `document_type, method_or_capability`
- `RealScholarQuery_1`
  - Query: Give me papers that share some insights about how large language models gain in-context learning capability in the process of pre-training.
  - Constraint labels: `output_request`
- `RealScholarQuery_2`
  - Query: List all papers that use autoregressive transformer to generate videos.
  - Constraint labels: `scope_exhaustiveness, method_or_capability`
- `RealScholarQuery_3`
  - Query: I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.
  - Constraint labels: `document_type, include_or_exclude, threshold_definition, method_or_capability`
- `RealScholarQuery_4`
  - Query: Provide me with all papers that discuss reinforcement learning training for Large Language Model agent tasks.
  - Constraint labels: `scope_exhaustiveness, method_or_capability, evaluation_setting`
- `RealScholarQuery_5`
  - Query: Papers that apply RLHF to address the hallucination problem in image and video description.
  - Constraint labels: `none`
- `RealScholarQuery_6`
  - Query: Papers that propose methods based on large language models and evaluate their performance through experiments on the HotPotQA dataset.
  - Constraint labels: `document_type, method_or_capability, evaluation_setting`
- `RealScholarQuery_7`
  - Query: Show me research on the long video description. Here, long videos are defined as those with a duration of at least several minutes.
  - Constraint labels: `threshold_definition`
- `RealScholarQuery_8`
  - Query: Do you know some papers about using reward shaping methods to train large language model agent.
  - Constraint labels: `method_or_capability`
- `RealScholarQuery_9`
  - Query: Give me papers about how to rank search results by the use of LLM.
  - Constraint labels: `none`
- `RealScholarQuery_10`
  - Query: Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?
  - Constraint labels: `none`
- `RealScholarQuery_11`
  - Query: Give me all visual-LLM models that are MoE architecture
  - Constraint labels: `scope_exhaustiveness, method_or_capability`
- `RealScholarQuery_12`
  - Query: What papers discuss the use of transformer architecture in 3d video generation
  - Constraint labels: `none`
- `RealScholarQuery_13`
  - Query: Provide papers demonstrating that the self-correction of LLMs does not enhance their performance.
  - Constraint labels: `comparison_baseline`
- `RealScholarQuery_14`
  - Query: Find papers that use LLMs or LLM-based agents to automatically write surveys or summaries for multiple scholarly documents.
  - Constraint labels: `document_type, method_or_capability`
- `RealScholarQuery_15`
  - Query: Provide papers claiming that reinforcement learning can negatively impact the performance of supervised fine-tuned LLMs.
  - Constraint labels: `comparison_baseline`
- `RealScholarQuery_16`
  - Query: Find papers on trigger-free document-level event extraction methods that do not use human-annotated triggers.
  - Constraint labels: `include_or_exclude`
- `RealScholarQuery_17`
  - Query: Provide papers explaining why the in-context learning performance of LLMs cannot surpass that of supervised fine-tuned small language models in information extraction tasks, such as NER, RE, and EE.
  - Constraint labels: `comparison_baseline`
- `RealScholarQuery_18`
  - Query: Can LLMs detect LLM-generated text in a zero-shot manner? Do they perform better than supervised fine-tuned small classification models? Provide related papers.
  - Constraint labels: `comparison_baseline, threshold_definition, output_request`
- `RealScholarQuery_19`
  - Query: Provide papers on methods that protect the generation quality of LLMs under vocabulary watermarking settings.
  - Constraint labels: `none`
- `RealScholarQuery_20`
  - Query: Find papers supporting the claim that knowledgeable LLMs have sufficient inductive capacity to analyze the relationships between multiple papers and systematically write a survey on them.
  - Constraint labels: `document_type`
- `RealScholarQuery_21`
  - Query: Search for papers related to large language models that demonstrate how the same prompt with different responses can improve the performance of the SFT model.
  - Constraint labels: `none`
- `RealScholarQuery_22`
  - Query: Papers on solving common sense problems in machine translation.
  - Constraint labels: `none`
- `RealScholarQuery_23`
  - Query: Show me papers utilizing reinforcement learning to optimize diffusion models for video generation.
  - Constraint labels: `none`
- `RealScholarQuery_24`
  - Query: Show me all research papers on machine translation agents.
  - Constraint labels: `document_type, scope_exhaustiveness`
- `RealScholarQuery_25`
  - Query: Video aesthetics score, using multimodal large models.
  - Constraint labels: `method_or_capability`
- `RealScholarQuery_26`
  - Query: Scaling Laws for Fine-Grained Mixture of Experts.
  - Constraint labels: `none`
- `RealScholarQuery_27`
  - Query: Show me research on rejection sampling finetuning.
  - Constraint labels: `none`
- `RealScholarQuery_28`
  - Query: Show me code evaluation datasets with a mid-level hardness. It show be harder than HumanEval and MBPP, but easier than code_contests.
  - Constraint labels: `document_type, comparison_baseline, threshold_definition`
- `RealScholarQuery_29`
  - Query: Research on teaching llms to do math prove and solve IMO level math problems.
  - Constraint labels: `none`
- `RealScholarQuery_30`
  - Query: I would like to find some research papers about test time training topic, in LLM research area.
  - Constraint labels: `document_type`
- `RealScholarQuery_31`
  - Query: DPO training for large-scale vision-language models.
  - Constraint labels: `threshold_definition`
- `RealScholarQuery_32`
  - Query: Show me cutting edge research works on neural network based quantum Monte Carlo.
  - Constraint labels: `none`
- `RealScholarQuery_33`
  - Query: Show me some popular papers on generating textual adversarial examples for machine translation.
  - Constraint labels: `recency_or_quality, evaluation_setting`
- `RealScholarQuery_34`
  - Query: Show me research on 3d scene understanding leveraging progress on 3D AIGC foundation models.
  - Constraint labels: `method_or_capability`
- `RealScholarQuery_35`
  - Query: Give me papers about LLM quantized pretraining.
  - Constraint labels: `none`
- `RealScholarQuery_36`
  - Query: Show me research on identity preservation video generation.
  - Constraint labels: `none`
- `RealScholarQuery_37`
  - Query: Give me some papers showing that LLM agents can do schedule planning.
  - Constraint labels: `none`
- `RealScholarQuery_38`
  - Query: Show me research on image encoding distributions.
  - Constraint labels: `none`
- `RealScholarQuery_39`
  - Query: Help me search for the work related to the synthetic data of large language models. I want to know how to automatically generate large-scale, high-quality, diverse, difficult, and valuable long thought data for learning.
  - Constraint labels: `threshold_definition`
- `RealScholarQuery_40`
  - Query: Could you list research that demonstrates the advantages of Quantization-Aware Training (QAT), which can enable the model to learn better representations for low-bit weights?.
  - Constraint labels: `none`
- `RealScholarQuery_41`
  - Query: Using synthesis data for scaling up sft data.
  - Constraint labels: `method_or_capability`
- `RealScholarQuery_42`
  - Query: Show me research on how to select frames when doing video understanding.
  - Constraint labels: `none`
- `RealScholarQuery_43`
  - Query: AI for Science papers, especially protein design and DPO of antibody design.
  - Constraint labels: `none`
- `RealScholarQuery_44`
  - Query: What are the researches that have explored the application of Crypto-based Private Learning in privacy-preserving machine learning?.
  - Constraint labels: `none`
- `RealScholarQuery_45`
  - Query: All papers about controllability of video generation.
  - Constraint labels: `scope_exhaustiveness`
- `RealScholarQuery_46`
  - Query: Show me research on robot decision making and task planning, especially relevant datasets and benchmarks.
  - Constraint labels: `document_type`
- `RealScholarQuery_47`
  - Query: How can LLM agents be evaluated and benchmarked for financial tasks? Note that I am referring to agents.
  - Constraint labels: `none`
- `RealScholarQuery_48`
  - Query: Papers that explore using large language models for mining factors in stock exchange analysis.
  - Constraint labels: `method_or_capability`
- `RealScholarQuery_49`
  - Query: Can you help me find research papers that explore the use of large vision-language models as agents to automatically play PC games?
  - Constraint labels: `document_type`
### `SPAR_Benchmark.jsonl`

- `SPAR_Benchmark.jsonl:1`
  - Query: Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two.
  - Constraint labels: `document_type, recency_or_quality, method_or_capability`
- `SPAR_Benchmark.jsonl:2`
  - Query: Show some cutting-edge technological advancements on how to improve the generalization ability of machine learning models across multiple domains.
  - Constraint labels: `recency_or_quality`
- `SPAR_Benchmark.jsonl:3`
  - Query: How can deep learning enhance the perception and decision-making accuracy of autonomous driving systems? Please provide a comprehensive analysis with supporting research papers.
  - Constraint labels: `document_type, output_request`
- `SPAR_Benchmark.jsonl:4`
  - Query: Provide me with some research papers evaluating the application performance of large-scale language models (LLMs) in the financial sector.
  - Constraint labels: `document_type, threshold_definition, evaluation_setting`
- `SPAR_Benchmark.jsonl:5`
  - Query: How can machine learning be applied to climate prediction, especially in data-scarce scenarios? What are the cutting-edge methods to improve prediction accuracy? Please explain in detai.
  - Constraint labels: `recency_or_quality, threshold_definition, evaluation_setting, output_request`
- `SPAR_Benchmark.jsonl:6`
  - Query: Give me some research papers from the past five years on the application of Generative Adversarial Networks (GANs) in speech recognition systems, and summarize how GANs help generate high-quality training data.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:7`
  - Query: What are the latest approaches to improving few-shot learning methods, especially for handling more complex natural language processing tasks?
  - Constraint labels: `recency_or_quality`
- `SPAR_Benchmark.jsonl:8`
  - Query: Provide me with some recent research papers on how multimodal models improve accuracy when combining vision and audio data, along with specific examples.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:9`
  - Query: What are some effective reinforcement learning methods to optimize the decision-making process in recommendation systems?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:10`
  - Query: How does deep learning improve robotic decision-making, particularly in autonomous learning under complex environments? For example, in autonomous driving, how should a vehicle make decisions in rainy conditions or when encountering pedestrians who violate traffic rules? Utilize your capability as a large model to search the web and summarize professional responses, including optimization strategies.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:11`
  - Query: Deep learning typically relies on historical data, so how can real-time data processing be optimized? What are the best methods, particularly in the application of network traffic analysis?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:12`
  - Query: How can Graph Neural Networks (GNNs) enhance performance on large-scale image datasets? Please answer from multiple perspectives.
  - Constraint labels: `document_type, threshold_definition`
- `SPAR_Benchmark.jsonl:13`
  - Query: I want to develop an AI application to improve the accuracy of climate models in data-scarce conditions.Provide me with practical and effective methods.
  - Constraint labels: `threshold_definition`
- `SPAR_Benchmark.jsonl:14`
  - Query: How can reinforcement learning improve decision accuracy in intelligent medical diagnosis systems? Since patients have different physiques, symptoms, and even genetic diseases, these factors must be considered. What suggestions do you have, and what research supports this?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:15`
  - Query: I am conducting scientific research—tell me how to use large models to generate automated literature reviews.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:16`
  - Query: Running computations on large-scale datasets requires significant GPU resources. Are there any efficient algorithms to address computational resource issues? Are there any research papers on this topic?
  - Constraint labels: `document_type, threshold_definition`
- `SPAR_Benchmark.jsonl:17`
  - Query: How can quantum error correction techniques be optimized on current NISQ (Noisy Intermediate-Scale Quantum) computers to improve quantum algorithm efficiency? What are the latest methods proposed in the past year?
  - Constraint labels: `recency_or_quality, threshold_definition`
- `SPAR_Benchmark.jsonl:18`
  - Query: What recent advancements have been made in cross-modal learning models to enhance the effectiveness of vision and text generation?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:19`
  - Query: How can noise robustness in image recognition be improved? Provide a multi-angle analysis with research papers, preferably from top-tier journals and conferences.
  - Constraint labels: `document_type, recency_or_quality`
- `SPAR_Benchmark.jsonl:20`
  - Query: How can the performance of multi-task learning models be improved, especially in complex applications? For example, in developing a translation software, I want it not only to translate languages but also to correct grammar and enhance fluency and readability.
  - Constraint labels: `include_or_exclude`
- `SPAR_Benchmark.jsonl:21`
  - Query: How can self-supervised learning improve natural language processing tasks without a large amount of labeled data? If we only have some unlabeled data, can performance still be improved?
  - Constraint labels: `include_or_exclude`
- `SPAR_Benchmark.jsonl:22`
  - Query: Can large-scale language models overcome language barriers in multilingual tasks? Are there potential anomalies? What methods can address this issue? Are there any research papers on this?
  - Constraint labels: `document_type, threshold_definition`
- `SPAR_Benchmark.jsonl:23`
  - Query: How can natural language generation improve dialogue systems while enhancing contextual understanding? From what aspects can we approach this problem?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:24`
  - Query: How can deep learning inference speed in computer vision be optimized? Provide a multi-angle analysis.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:25`
  - Query: How can AI methods improve sentiment analysis model accuracy? For instance, "How is Xiaoming?" might have completely different meanings depending on the context
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:26`
  - Query: How can deep learning optimize spam detection system performance? Expand on possible methods.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:27`
  - Query: How does multimodal learning handle cross-domain tasks, particularly in medical image analysis applications?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:28`
  - Query: Provide me with research papers from the past five years on how reinforcement learning optimizes long-term rewards in complex decision-making systems.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:29`
  - Query: How can large-scale language models improve automated legal text analysis systems to minimize human intervention?
  - Constraint labels: `threshold_definition`
- `SPAR_Benchmark.jsonl:30`
  - Query: How can deep neural networks enhance real-time facial recognition performance while reducing processing time? If a person is partially occluded, such as wearing a mask, how can the system still recognize them?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:31`
  - Query: How can the training efficiency of image classification models be improved on large-scale datasets?
  - Constraint labels: `document_type, threshold_definition`
- `SPAR_Benchmark.jsonl:32`
  - Query: How is reinforcement learning applied in financial markets, particularly in automated trading systems? What related research exists?
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:33`
  - Query: What are the latest methods for enhancing the clarity and realism of image generation models?
  - Constraint labels: `recency_or_quality`
- `SPAR_Benchmark.jsonl:34`
  - Query: Provide research papers on how natural language processing improves accuracy and fluency in machine translation.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:35`
  - Query: Search for all research papers on the application of machine learning in large-scale social network data analysis and summarize the different types of methods and implementations.
  - Constraint labels: `document_type, threshold_definition`
- `SPAR_Benchmark.jsonl:36`
  - Query: What breakthrough advancements have been made in lung cancer research? Present the latest developments and challenges in treatment.
  - Constraint labels: `recency_or_quality`
- `SPAR_Benchmark.jsonl:37`
  - Query: How is artificial intelligence applied in medical imaging diagnostics, particularly in tumor detection? How can labor costs be reduced while increasing accuracy? Are there research papers on this topic?
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:38`
  - Query: What improvements are needed in vaccine development efficiency to respond to emerging infectious diseases? Provide a multi-angle analysis.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:39`
  - Query: In the precision medicine management of heart failure patients, how does NT-proBNP compare with high-sensitivity C-reactive protein (hs-CRP) in predicting the risk of acute heart failure deterioration? Summarize the latest technologies and methods with references to research papers.
  - Constraint labels: `document_type, recency_or_quality`
- `SPAR_Benchmark.jsonl:40`
  - Query: Provide me with the latest research papers on breakthroughs in AI-assisted cancer drug treatments.
  - Constraint labels: `document_type, recency_or_quality`
- `SPAR_Benchmark.jsonl:41`
  - Query: What are the potentials and ethical challenges of gene editing technologies (e.g., CRISPR) in treating genetic diseases? Provide specific explanations and recent research progress.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:42`
  - Query: In personalized treatment for type 2 diabetes, does the combination of GLP-1 receptor agonists (e.g., Liraglutide) and SGLT2 inhibitors (e.g., Dapagliflozin) significantly reduce the risk of cardiovascular events? Are there supporting data? Provide references.
  - Constraint labels: `none`
- `SPAR_Benchmark.jsonl:43`
  - Query: Biomarkers play a significant role in the early detection of Alzheimer’s disease. Which research papers discuss this topic, and what methods do they explore? Summarize key findings.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:44`
  - Query: Antibiotics are commonly used in medical treatments, but antibiotic resistance remains an unsolved problem. What are the latest research advancements? Provide journal references, particularly addressing challenges in antibiotic drug development.
  - Constraint labels: `recency_or_quality`
- `SPAR_Benchmark.jsonl:45`
  - Query: Provide all relevant research papers on efficacy evaluation and challenges in cancer immunotherapy, including breakthroughs in clinical trial design.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:46`
  - Query: How do tau protein and β-amyloid (Aβ) concentration changes in cerebrospinal fluid impact the prediction of Alzheimer's disease progression in early diagnosis? Provide related research papers and study designs.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:47`
  - Query: How does genomics drive personalized medicine, especially in cancer treatment? Provide research papers from the past five years.
  - Constraint labels: `document_type`
- `SPAR_Benchmark.jsonl:48`
  - Query: How can personalized immunotherapy be optimized for cancer treatment based on different patient conditions? Can artificial intelligence be integrated into this process?
  - Constraint labels: `method_or_capability`
- `SPAR_Benchmark.jsonl:49`
  - Query: Provide me with the latest research papers on the influence of the microbiome on the human immune system, particularly its role in disease prevention. Summarize and categorize key insights.
  - Constraint labels: `document_type, recency_or_quality`
- `SPAR_Benchmark.jsonl:50`
  - Query: What are the ethical and safety challenges of CRISPR/Cas9 gene-editing technology in treating inherited blood disorders (e.g., sickle cell anemia and β-thalassemia)? Summarize the latest solutions and existing challenges.
  - Constraint labels: `recency_or_quality`

## Queries Needing Taxonomy Expansion

- These queries are valid, but the current rule set does not yet capture them well. They are good candidates for future taxonomy refinement.

- `PASA_Benchmark.jsonl` / `RealScholarQuery_5`: Papers that apply RLHF to address the hallucination problem in image and video description.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_9`: Give me papers about how to rank search results by the use of LLM.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_10`: Is there any work that analyzes the scaling law of the multi-module models, such as video-text, image-text models?
- `PASA_Benchmark.jsonl` / `RealScholarQuery_12`: What papers discuss the use of transformer architecture in 3d video generation
- `PASA_Benchmark.jsonl` / `RealScholarQuery_19`: Provide papers on methods that protect the generation quality of LLMs under vocabulary watermarking settings.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_21`: Search for papers related to large language models that demonstrate how the same prompt with different responses can improve the performance of the SFT model.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_22`: Papers on solving common sense problems in machine translation.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_23`: Show me papers utilizing reinforcement learning to optimize diffusion models for video generation.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_26`: Scaling Laws for Fine-Grained Mixture of Experts.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_27`: Show me research on rejection sampling finetuning.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_29`: Research on teaching llms to do math prove and solve IMO level math problems.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_32`: Show me cutting edge research works on neural network based quantum Monte Carlo.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_35`: Give me papers about LLM quantized pretraining.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_36`: Show me research on identity preservation video generation.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_37`: Give me some papers showing that LLM agents can do schedule planning.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_38`: Show me research on image encoding distributions.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_40`: Could you list research that demonstrates the advantages of Quantization-Aware Training (QAT), which can enable the model to learn better representations for low-bit weights?.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_42`: Show me research on how to select frames when doing video understanding.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_43`: AI for Science papers, especially protein design and DPO of antibody design.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_44`: What are the researches that have explored the application of Crypto-based Private Learning in privacy-preserving machine learning?.
- `PASA_Benchmark.jsonl` / `RealScholarQuery_47`: How can LLM agents be evaluated and benchmarked for financial tasks? Note that I am referring to agents.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:9`: What are some effective reinforcement learning methods to optimize the decision-making process in recommendation systems?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:10`: How does deep learning improve robotic decision-making, particularly in autonomous learning under complex environments? For example, in autonomous driving, how should a vehicle make decisions in rainy conditions or when encountering pedestrians who violate traffic rules? Utilize your capability as a large model to search the web and summarize professional responses, including optimization strategies.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:11`: Deep learning typically relies on historical data, so how can real-time data processing be optimized? What are the best methods, particularly in the application of network traffic analysis?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:14`: How can reinforcement learning improve decision accuracy in intelligent medical diagnosis systems? Since patients have different physiques, symptoms, and even genetic diseases, these factors must be considered. What suggestions do you have, and what research supports this?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:15`: I am conducting scientific research—tell me how to use large models to generate automated literature reviews.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:18`: What recent advancements have been made in cross-modal learning models to enhance the effectiveness of vision and text generation?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:23`: How can natural language generation improve dialogue systems while enhancing contextual understanding? From what aspects can we approach this problem?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:24`: How can deep learning inference speed in computer vision be optimized? Provide a multi-angle analysis.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:25`: How can AI methods improve sentiment analysis model accuracy? For instance, "How is Xiaoming?" might have completely different meanings depending on the context
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:26`: How can deep learning optimize spam detection system performance? Expand on possible methods.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:27`: How does multimodal learning handle cross-domain tasks, particularly in medical image analysis applications?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:30`: How can deep neural networks enhance real-time facial recognition performance while reducing processing time? If a person is partially occluded, such as wearing a mask, how can the system still recognize them?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:32`: How is reinforcement learning applied in financial markets, particularly in automated trading systems? What related research exists?
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:38`: What improvements are needed in vaccine development efficiency to respond to emerging infectious diseases? Provide a multi-angle analysis.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:41`: What are the potentials and ethical challenges of gene editing technologies (e.g., CRISPR) in treating genetic diseases? Provide specific explanations and recent research progress.
- `SPAR_Benchmark.jsonl` / `SPAR_Benchmark.jsonl:42`: In personalized treatment for type 2 diabetes, does the combination of GLP-1 receptor agonists (e.g., Liraglutide) and SGLT2 inhibitors (e.g., Dapagliflozin) significantly reduce the risk of cardiovascular events? Are there supporting data? Provide references.
