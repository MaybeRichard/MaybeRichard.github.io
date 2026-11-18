---
title: 扩散模型合成图像用于数据增强的研究脉络与方法比较
date: 2026-11-18
summary: 基于 Zotero 中 87 篇纳入文献的个人阅读综述，梳理从 GAN 到扩散模型、从图像生成到任务反馈和数据集蒸馏的演进，并讨论结果可比性、失败模式与开源情况。
tags: [扩散模型, 合成数据, 数据增强, 计算机视觉, 综述]
tldr: 这批文献共同指向一个结论：合成数据的价值取决于它是否补上了下游模型真正缺失的变化因素。生成得更逼真只是必要条件，样本选择、标签一致性、分布覆盖和训练方式决定了最终收益。
authors: [MaybeRichard]
references:
  gan2014:
    author: Goodfellow et al.
    year: 2014
    text: "Goodfellow et al. Generative Adversarial Nets."
    url: https://arxiv.org/abs/1406.2661
  ddpm2020:
    author: Ho et al.
    year: 2020
    text: "Ho et al. Denoising Diffusion Probabilistic Models."
    url: https://arxiv.org/abs/2006.11239
  ldm2022:
    author: Rombach et al.
    year: 2022
    text: "Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models."
    url: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
  controlnet2023:
    author: Zhang et al.
    year: 2023
    text: "Zhang et al. Adding Conditional Control to Text-to-Image Diffusion Models."
    url: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
  unidiffda2026:
    author: Li et al.
    year: 2026
    text: "Li et al. Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation."
    url: https://arxiv.org/abs/2603.08364
  tadasurvey2026:
    author: When More Data Hurts
    year: 2026
    text: "When More Data Hurts: A Critical Survey of Failure Modes in Generative Data Augmentation for Image Classification."
    url: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6448800
---

## 1. 先说结论

扩散模型带来的改变，不只是把 GAN 换成了另一种生成器。研究重点经历了三个转移。

第一阶段关注“能不能生成足够真实的新图像”。第二阶段关注“怎样保留标签、细节和结构，同时增加变化”。最近的工作进一步把下游分类器、检测器或分割器接入生成过程，尝试生成真正能改变模型决策的样本。

从已有证据看，合成数据最稳定的角色是补充真实数据。医学影像系统综述显示，合成数据加入真实训练集通常带来小幅分类收益，分割收益更小。synthetic-only 训练通常落后于真实数据训练。生成数量增加会出现边际收益下降，选择、过滤和任务感知比单纯扩大数量更重要。

## 2. 发展脉络

### 2.1 2023：从生成图像走向生成训练数据

Azizi 等人的工作把类条件扩散模型用于 ImageNet 数据生成，证明合成图像能够支持分类器训练。GIF 将问题表述为 dataset expansion，用少量真实图像引导 DALL-E 2 或 Stable Diffusion 生成新的带标签样本，同时指出 synthetic-only 仍然不如同规模真实数据。

DiffuMask 和 DatasetDM 将扩增对象从图像扩展到图像与感知标注。DiffuMask 观察文本条件扩散中的 cross-attention，利用它自动得到像素级语义掩码。DatasetDM 在扩散模型的 latent 上加入感知解码器，可以生成图像、语义分割、实例分割和深度等配对数据。论文报告的代表性结果是 VOC 语义分割提升 13.3 个 mIoU，COCO 实例分割提升 12.1 个 AP。StableRep 则把合成数据用于视觉表示学习，将同一个文本提示生成的多个图像作为正样本进行多正例对比学习。

### 2.2 2024：控制保真度、多样性和少样本适配

DA-Fusion 从真实图像编辑起点开展增广，起点位于真实图像而非纯噪声，使用文本变化描述控制颜色、材质和场景变化。这样可以更好地保留原图结构，也能处理预训练模型没有直接见过的概念。

DiffuseMix 使用条件提示重塑输入图像，再通过拼接和分形混合构造增广结果。Diff-Mix 进一步利用跨类别图像插值增加变化，明确处理 faithfulness 与 diversity 的平衡。DataDream 用少量真实样本对文本编码器和 U-Net 做 LoRA 微调，分别提供全数据集和逐类别的适配方式。它说明在 few-shot 条件下，轻量适配可以明显缩小生成分布和目标分布之间的差距。

这一阶段还出现了两条并行路线。DiffAug 研究 diffuse-and-denoise 增广对鲁棒分类的作用，SynCLR 和 StableRep 把重点放在合成数据训练视觉表示。扩散模型开始从下游训练集的“附加样本生成器”变成预训练数据源。

### 2.3 2025—2026：任务感知、长尾和医学场景

Diff-II 把扩散反演后的潜变量放在一个插值圆上，通过两阶段去噪同时提高生成样本的保真度和多样性。MRCA 让实例分割器反过来指导扩散模型，多轮生成模型不确定的对象，并按类别预算把对象放回真实图像。DBG 则围绕决策边界生成样本，并用分类器清洗有害样本。CORAL 发现长尾类别在 U-Net latent 中可能与头部类别重叠，于是用对比正则化解耦类别表示。DAR-GDA 通过蒸馏、对抗对齐和重要性重加权同时解决采样效率和分布偏移。

医疗和稠密预测工作更加关注结构一致性。MatchMask 从少量语义掩码生成图像—掩码对，用 Gradient Probe 找到关键层，再用轻量适配器和相对过滤抑制错误区域。MRGen 以文本和掩码共同控制 MRI 生成，服务于缺少标注的模态。Dual-Stream Diffusion 将结构流和外观流分开去噪，再用时间门控融合。Polyp-LDM、DiDGen、DiffAtlas、医学 DiffAug 和 MRGen 都把图像质量和下游分割效果同时作为评价目标。

2026 年的一个重要变化是评测开始系统化。UniDiffDA 将方法拆成模型微调、样本生成和样本利用三个阶段，在通用、细粒度、长尾、跨域和医学分类上重新实现多个方法。结论呈现出明显的任务依赖性，三阶段组合与目标任务共同决定结果。这个视角比单一榜单更适合解释论文之间为何会得到不同结论。

## 3. 方法分类

### 3.1 冻结生成器加提示

最简单的方案是用类别名、属性或文本描述提示预训练 T2I 模型。优点是无需目标域微调，能快速扩大数据覆盖。缺点是提示中的语义往往比类别标签更粗，细粒度类别容易出现背景捷径、姿态偏差和类别混淆。SynCLR、StableRep、Fake It Till You Make It 说明这种方案可以支撑表示学习，但结果高度依赖提示模板、生成模型和合成规模。

### 3.2 图像条件编辑

输入一张真实图像，先反演或加噪，再在去噪过程中控制变化。它更容易保留标签和布局，适合分类增广和鲁棒性训练。DA-Fusion、DiffuseMix、Diff-II、DiffAug 和 1S-DAug 属于这一类。其主要代价是新颖性受输入图像限制，生成结果可能只是原图的低幅变化。

### 3.3 LoRA、Textual Inversion 和轻量适配

少样本场景通常需要一定程度的领域适配。DataDream、LoFT、MedDiff-FT、BOB 和 ChimeraLoRA 通过 LoRA 或文本嵌入学习类别和实例信息。BOB 额外提取背景和姿态等类别无关属性，在微调时显式建模，生成时再对这些属性边缘化。在 Aircraft 的五样本设置中，BOB 将分类准确率从 50.0% 提高到 57.4%。ChimeraLoRA 将类共享适配器和逐图像适配器分开，分别保存类别先验和实例细节。

### 3.4 任务模型反馈

如果目标是改善分类器，就用分类器的梯度、边界或不确定性告诉生成器“还缺什么”。DBG 生成决策边界附近的样本，MRCA 让实例分割器寻找不确定目标，C2I 用分类梯度一致性构造奖励，S3OD 用扩散特征、概念注意力和 DINO 特征共同生成显著目标。它们通常比无目标的随机增广更有效，代价是反馈模型的偏差会进入生成过程。

### 3.5 长尾平衡和分布对齐

长尾任务不能只复制尾类。CORAL 在扩散 latent 中分离重叠的头尾类别表示，SynBalance 先合成再选择互补的少量样本，DAR-GDA 用对抗对齐和重要性重加权修正生成分布，Diffusing DeBias 利用扩散模型的偏置放大机制为去偏模型构造辅助数据。它们共同说明，类别数量平衡仍需结合决策空间覆盖来判断。

### 3.6 图像—标注联合生成

分割、检测和深度估计要求图像与标注保持一致。DiffuMask 通过注意力得到 mask，DatasetDM 用感知解码器生成多种标注，MatchMask、MRGen、DSD、Polyp-LDM、DiDGen、DiffAtlas、D3S2 和 S2-RL 则直接用 mask、layout、sketch 或双流扩散建模结构。对这些任务来说，FID 和人眼质量主要反映图像外观，标签可用性仍需单独验证。

### 3.7 生成后筛选和数据集蒸馏

生成后筛选成为降低噪声和成本的独立模块。Chamfer Guidance 用少量真实样本估计质量与多样性，Post-Generation Curation 将样本分成 homogeneous 和 heterogeneous 部分再打分，BRIDGE 用训练集内部的深度近邻距离寻找欠表示区域。IMS3、EVLF、D3S2、S2-RL、DIVER 和 SFM 则更进一步，把目标变成可用的极小合成数据集或蒸馏数据。

## 4. 结果如何比较

论文报告的指标需要在统一实验设置下比较，跨数据集结果不宜拼成一张总榜。分类工作通常报告 Top-1、balanced accuracy、tail accuracy 或 OOD accuracy。检测工作报告 AP、召回率和跨域表现。分割工作报告 mIoU、Dice 或边界指标。生成质量常见 FID、KID、CLIP 相似度和人评，但它们与下游效用不总是一致。

比较时至少需要固定以下因素。

1. 真实数据量和合成数据比例。
2. 生成器是从头训练、领域微调还是完全冻结。
3. 文本、类别、图像、mask、layout 等控制信号。
4. 采样步数、guidance scale、生成分辨率和筛选预算。
5. 下游模型结构、训练轮数、增强策略和随机种子。
6. 是否把合成数据单独训练，还是与真实数据混合。
7. 是否报告每个类别、跨域和外部测试集的表现。

UniDiffDA 的价值在于把这些变量拆开评估。医学系统综述则从更高层面提醒，合成数据的增益主要来自与真实数据互补，真实数据仍然承担核心监督作用。

## 结果比较：几个可以落地的数字

下面的数字来自论文各自的实验设置，适合用来理解方法作用，跨数据集排行榜需要额外的统一实验。

| 方法或证据 | 任务 | 论文报告的结果 | 主要解释 |
|:--|:--|:--|:--|
| TADA | 分类 | 只增强训练早期未学到的 30%–40% 样本，最高提升 2.8 个百分点 | 选择样本可能比增加数量更重要 |
| BOB | 细粒度分类 | 24 个设置中 18 个超过先前方法，14 个设置提升超过 2 个百分点 | 显式建模背景和姿态等类无关因素 |
| MatchMask | 语义分割 | PASCAL VOC 从 67.5% 提升到 74.3% mIoU | mask 中心生成、层自适应融合和相对过滤共同作用 |
| DatasetDM | 分割和实例分割 | VOC 提升 13.3 mIoU，COCO 提升 12.1 AP | 感知解码器生成图像与标注配对数据 |
| S3OD | 显著性检测 | synthetic-only 的跨数据集错误减少 20%–50% | 合成场景和歧义感知架构共同改善跨域泛化 |
| MRCA | 长尾实例分割 | 约 6% 的生成预算仍能获得持续收益 | 用实例分割器反馈生成不确定目标 |
| 医学 DiffAug | 医学分割 | 小息肉和扁平病灶的 false negative rate 最高降低 28%，推理约 12.5 samples/s | 文本引导、质量验证和快速潜变量估计共同影响结果 |
| 医学影像系统综述 | 多种医学任务 | synthetic-only 通常有限，real+synthetic 的分类收益较小，分割收益更小 | 合成数据更像真实数据的补充 |

## 5. 局限和风险

**语义和分布偏移。** 大型自然图像模型并不自动理解医学形态、传感器噪声或三维解剖结构。提示错误会生成漂亮但标签不对的图像。

**保真度和多样性的冲突。** 强图像条件提高标签稳定性，却可能只复制训练集。弱条件增加变化，却会带来背景捷径和类别混淆。

**反馈闭环偏差。** 分类器、分割器或奖励模型既负责挑样本又影响生成器，可能把自身错误反复放大。

**计算和可复现性。** 反演、微调、多步采样、过滤和多轮反馈的成本很高。许多论文没有完整报告 prompt、随机种子、合成比例、过滤阈值和生成预算。

**隐私和记忆。** 预训练扩散模型可能复现训练数据中的人物、版权内容或敏感模式。合成图像也可能把数据集偏见放大到下游模型。

**外部验证不足。** 多数工作只在少量公开数据集和单一下游骨干上验证，跨医院、跨设备、跨地域的证据仍然有限。

## 6. 开源状况

当前 87 条记录中，有 47 条正文出现 GitHub URL。这个数字包含依赖库和基础模型链接，因此需要进一步区分作者仓库、依赖项目和基础模型。能够明确识别为作者仓库的代表性项目包括 DiffDA-Eval、DBG、IMS3、DisCL、Diff-II、CORAL、DAR-GDA、DiffusingDeBias、MedDiff-FT、DataDream、DA-Fusion、DatasetExpansion、DIAGen、MGRAttn-RNO、D3S2、LDMLR、SynRep/SynCLR、TRCG、SFD-Polyp、Polyp-LDM、DiDGen、DiffAtlas、DiGDA、LoFT、S2RL 和 SFM。权重、生成数据和训练脚本并不总是同时提供。

## 7. 仍需补全文本的条目

以下五条目前在 Zotero 中没有本地 PDF，文章中的方法和结果依据题目、摘要或出版页面，属于待补全文献。

- Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data
- Class-Mixed Diffusion Augmentation for Shortcut-Breaking in Continual Learning
- ADuLTS: Appearance Descriptions under Long-Tailed Scenarios with Diverse Synthesized Images
- Synthetic Imagery Improves Ecological Classification When Real Data Are Scarce, Not Direct Substitutes
- Triple-Prompt Controllable Diffusion for Universal Data Augmentation in Medical Image Segmentation

## 8. 一句话总结

扩散模型数据扩增正在从“生成更多图像”转向“生成对特定模型和特定缺口有用的图像”。下一步的关键在于建立可复现的任务效用评测、可靠的样本筛选、结构一致的标注生成和跨域外部验证。

## 逐篇证据索引

这部分覆盖本次纳入的 87 篇记录。它主要用于回到原文，不把自动提取的短摘要当作完整复述。

<details>
<summary>分类/识别（33篇）</summary>

- **Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation**（全文PDF已读取）  
  方法：In this work, we introduce UniDiffDA, a unified analytical framework that decomposes DiffDA methods into three core components: model fine-tuning, sample generation, and sample utilization.  
  结果：In this work, we introduce UniDiffDA, a unified analytical framework that decomposes diffusion-based data augmentation into three core components: model fine-tuning, sample generation, and sample utilization. Using this framework, we establish a com- prehensiv…  
  开源：[https://github.com/nukezil/DiffDA-Eval](https://github.com/nukezil/DiffDA-Eval)；[论文来源](https://arxiv.org/abs/2603.08364)
- **Decision Boundary-aware Generation for Long-tailed Learning (DBG)**（全文PDF已读取）  
  方法：To address this, we first iden- tify the problem of boundary ambiguity and then propose Decision Boundary-aware Generation (DBG) framework, which promotes near-boundary representation learning by generating informative n…  
  结果：Long-tailed recognition is hindered not only by data scarcity but also by boundary ambiguity introduced by data augmentation and head-to-tail transfer in generative aug- mentation. We made boundary ambiguity measurable with three metrics: inter-class overlap d…  
  开源：[https://github.com/keepdigitalabc-svg/DBG.](https://github.com/keepdigitalabc-svg/DBG.)；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Decision_Boundary-aware_Generation_for_Long-tailed_Learning_CVPR_2026_paper.html)
- **ChimeraLoRA: Multi-Head LoRA-Guided Synthetic Datasets**（全文PDF已读取）  
  方法：To expose coherent class semantics in the shared LoRA A, we propose a semantic boosting by preserving class bound- ing boxes during training.  
  结果：In this work, we propose a multi-head LoRA guided method for synthetic dataset generation, called ChimeraLoRA. By separating the roles of two low-rank adapters, we use a class-shared adapter to encode class priors and per-image adapters to model instance-level…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ChimeraLoRA_Multi-Head_LoRA-Guided_Synthetic_Datasets_CVPR_2026_paper.html)
- **Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification (BOB)**（全文PDF已读取）  
  方法：We propose a ﬁne-tuning strategy BOB (Beyond OBjects) to mitigate these concerns for ﬁne-grained classi- ﬁcation.  
  结果：We introduce BOB as a ﬁne-tuning strategy for text-to- image (T2I) models that mitigates overﬁtting and address- ing the unique challenges of ﬁne-grained classiﬁcation. By leveraging more detailed captioning to extract class- agnostic background and pose infor…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Beyond_Objects_Contextual_Synthetic_Data_Generation_for_Fine-Grained_Classification_CVPR_2026_paper.html)
- **Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion (DisCL)**（全文PDF已读取）  
  方法：Introduction While existing machine learning approaches can train repre- sentation or discriminative models with promising general- *These authors contributed equally to this work.  
  结果：In this paper, we introduce DisCL, a novel paradigm de- signed to enhance model performance when dealing with low-quality or scarce data. DisCL effectively bridges the distribution gap between original and target data using a spec- trum of synthetic data, part…  
  开源：[https://github.com/tianyi-lab/DisCL](https://github.com/tianyi-lab/DisCL)；[论文来源](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html)
- **Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification (Diff-II)**（全文PDF已读取）  
  方法：In this paper, we analyze today’s diffusion-based DA methods, and argue that they cannot take account of both faithfulness and diver- sity, which are two critical keys for generating high-quality samples and boosting cla…  
  结果：In this work, we analyze current diffusion-based DA meth- ods from a unified perspective, finding that they can either only improve the faithfulness of synthetic samples or only improve their diversity. To take both faithfulness and di- versity into account, w…  
  开源：[https://github.com/HKUST-LongGroup/Diff-II.](https://github.com/HKUST-LongGroup/Diff-II.)；[论文来源](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Inversion_Circle_Interpolation_Diffusion-based_Image_Augmentation_for_Data-scarce_Classification_CVPR_2025_paper.html)
- **CORAL: Disentangling Latent Representations in Long-Tailed Diffusion**（全文PDF已读取）  
  方法：To address this, we propose COntrastive Regularization for Aligning Latents (CORAL), a contrastive latent alignment framework that leverages supervised contrastive losses to encourage well-separated latent class represen…  
  结果：Ensuring high-quality sample generation for tail classes of long-tailed datasets remains a major challenge. In addressing this challenge, we have revealed a previously unknown cause for the poor performance of DMs: the (U-Net) latent representations for the ta…  
  开源：[https://github.com/SankarLab/coral-lt-diffusion.](https://github.com/SankarLab/coral-lt-diffusion.)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a32539cb16274581a17e679f6046f4bf-Abstract-Conference.html)
- **Principled Long-Tailed Generative Modeling via Diffusion Models**（全文PDF已读取）  
  方法：This paper develops a theoretical framework for long-tailed learning via diffusion models through the lens of deep mutual learning.  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/pranoydas51/IGD-ML](https://github.com/pranoydas51/IGD-ML)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9ff5493d7d4a4b4002d67738237abd5b-Abstract-Conference.html)
- **Diffusing DeBias: Synthetic Bias Amplification for Model Debiasing**（全文PDF已读取）  
  方法：This paper addresses this problem by leveraging bias ampli- fication with generated synthetic data only: we introduce Diffusing DeBias (DDB), a novel approach acting as a plug-in for common methods of unsupervised model …  
  结果：In this work, we present Diffusing DeBias (DDB), a novel debiasing framework capable of learning the per-class bias-aligned training data distribution, leveraging a CDPM. Synthetic images sampled from the inferred biased distribution are used to train a Bias A…  
  开源：[https://github.com/Malga-Vision/DiffusingDeBias.](https://github.com/Malga-Vision/DiffusingDeBias.)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a982938f457eafffedc3205c46b4bac7-Abstract-Conference.html)
- **AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition**（全文PDF已读取）  
  方法：In this paper, we introduce AugGen, a self-contained synthetic augmentation technique.  
  结果：In this work, we introduced AugGen, a novel yet simple sampling approach that carefully conditions a generator using a discriminative model, both trained on a single real dataset, to generate augmented samples. By combining these synthetic samples with the ori…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html)
- **Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification**（全文PDF已读取）  
  方法：In this paper, we propose a two- stage synthetic data augmentation framework, leveraging pre-trained diffusion models for long-tailed food classiﬁca- tion.  
  结果：In this paper, we proposed a novel two-stage synthetic data augmentation framework using pre-trained diffusion models for long-tailed food image classiﬁcation. To ad- dress inter-class confusion, we introduced a confusing class selection strategy that identiﬁe…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2025W/MTF/html/Koh_Synthetic_Data_Augmentation_using_Pre-trained_Diffusion_Models_for_Long-tailed_Food_CVPRW_2025_paper.html)
- **DataDream: Few-shot Guided Dataset Generation**（全文PDF已读取）  
  方法：We pro- pose DataDream, a framework for synthesizing classification datasets that more faithfully represents the real data distribution when guided by few-shot examples of the target classes.  
  结果：In this paper, we studied the efficacy of leveraging the generative models for im- proving the image classification performance in few-shot scenarios. We proposed DataDream, a method to generate synthetic data with the guidance of few-shot samples, which are t…  
  开源：[https://github.com/ExplainableML/DataDream.](https://github.com/ExplainableML/DataDream.)；[论文来源](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08991.pdf)
- **Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model (Diff-Mix)**（全文PDF已读取）  
  方法：To tackle this challenge, we introduce an innovative inter-class data augmentation method known as Diff-Mix 1, which enriches the dataset by performing image translations between classes.  
  结果：In this work, we investigate two pivotal aspects, faithfulness and diversity, that are critical for the current state-of-the-art text-to-image generative models to enhance image classiﬁ- cation tasks. To achieve a more effective balance between these two aspec…  
  开源：[https://github.com/Zhicaiwww/Diff-Mix](https://github.com/Zhicaiwww/Diff-Mix)；[论文来源](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Enhance_Image_Classification_via_Inter-Class_Image_Mixup_with_Diffusion_Model_CVPR_2024_paper.html)
- **DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models**（全文PDF已读取）  
  方法：To address these limitations, we propose DIFFUSEMIX, a novel data augmentation technique that leverages a diffu- sion model to reshape training images, supervised by our bespoke conditional prompts.  
  结果：In this paper, we introduced DIFFUSEMIX, a data augmen- tation technique based on diffusion models to increase di- versity in the data while preserving the original seman- tics of the input image. It involves generation, concate- nation, and fractal blending s…  
  开源：正文未确认作者代码；[论文来源](https://doi.org/10.1109/cvpr52733.2024.02608)
- **Effective Data Augmentation With Diffusion Models (DA-Fusion)**（全文PDF已读取）  
  方法：Our method edits images to change their semantics using an off-the-shelf diffusion model, and generalizes to novel visual concepts from a few labelled examples.  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/brandontrabucco/da-fusion.](https://github.com/brandontrabucco/da-fusion.)、[https://github.com/rom1504/clip-retrieval](https://github.com/rom1504/clip-retrieval)；[论文来源](https://proceedings.iclr.cc/paper_files/paper/2024/file/3f1351a10a3f904e6f3845a94275a909-Paper-Conference.pdf)
- **Fill-Up: Balancing Long-Tailed Data with Generative Models**（全文PDF已读取）  
  方法：Through extensive experiments, this paper proposes a new image synthesis pipeline for long-tailed situ- ations using Textual Inversion.  
  结果：This paper introduces the concept of filling-up long-tailed data with synthetic samples from recent large-scale text-to-image synthesis models. We provide valuable insights into different methods of synthetic data generation through detailed experiments and pr…  
  开源：[https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)、[https://github.com/rinongal/textual_inversion/](https://github.com/rinongal/textual_inversion/)、[https://github.com/runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)；[论文来源](https://arxiv.org/abs/2306.07200)
- **Limitations of Synthetic Data Generation in Specialized Data-Scarce Domains**（全文PDF已读取）  
  方法：Con- trary to prevailing expectations, we find that diffusion-generated synthetic aug- mentation provides little to no improvement over conventional non-generative augmentation strategies.  
  结果：By investigating whether modern generative models can meaningfully improve classifier performance in extreme low-data regimes, we find that, across a diverse set of distribution modeling and perturbation-based synthetic data generation strategies, synthetic au…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2608.13729)
- **1S-DAug: One-Shot Data Augmentation for Robust Few-Shot Generalization**（全文PDF已读取）  
  方法：We first analyze the importance of test-time augmentation, and then design and study a simple yet natural operator named 1S-DAug, which comprises geometric perturbations with controlled noise injection and image-conditio…  
  结果：To conclude, we identify the scarcity of test-time data augmentation in the literature, and study the usefulness of one-shot test-time generative augmentation for generalization improvement. Our findings are positive: 1S-DAug, as a simple plug-in for frozen FS…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2602.00114)
- **Cross-Attention-Driven Propose-and-Select Generative Data Augmentation for Few-Shot Image Classification**（全文PDF已读取）  
  方法：To address these limitations, we propose a two-stage Propose-and- Select framework for controllable data augmentation.  
  结果：In this work, we have presented a novel two-stage Propose-and-Select framework to overcome the fundamental bottlenecks of unconstrained generative data augmentation in few-shot image classification. By transitioning from the conventional “generate-once, use-di…  
  开源：正文未确认作者代码；[论文来源](https://www.mdpi.com/1424-8220/26/14/4590)
- **Controlling Augmentations for Balancing Invariance and Diversification in One-Shot Image Classification**（全文PDF已读取）  
  方法：In this work, we propose an effective framework that integrates a large language model (LLM) with a DM through a controllable, two-type augmentation pro­ cess.  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/kkyuhun94/](https://github.com/kkyuhun94/)；[论文来源](https://doi.org/10.1016/j.neucom.2026.134322)
- **DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge**（全文PDF已读取）  
  方法：DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge Jie Shao Ke Zhu Hanxiao Zhang Jianxin Wu∗ National Key Laboratory for Novel Software Technology, Nanjing University, China School of Artificial Inte…  
  结果：In this research, we proposed a novel, data-centric approach designed to address the challenges of long-tail classification. We defined and identified AID (approximately in-distribution) samples as the important ones. We then revised a diffusion model trained …  
  开源：正文未确认作者代码；[论文来源](https://doi.org/10.52202/079017-3909)
- **Increasing the Utility of Synthetic Images through Chamfer Guidance**（全文PDF已读取）  
  方法：In this work, we introduce Chamfer Guidance: a training-free guidance approach which leverages a handful of real exemplar images to characterize the quality and diversity of synthetic data.  
  结果：We introduced Chamfer Guidance, a novel training-free approach to improve the utility of synthetic data from conditional image models. By leveraging a small set of exemplar real images to guide the generation process, our method balances quality and diversity …  
  开源：[https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)、[https://github.com/layer6ai-labs/dgm-eval](https://github.com/layer6ai-labs/dgm-eval)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cd08b0700a0a4f4856c70d9346b3a506-Abstract-Conference.html)
- **SaSPA: Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation**（全文PDF已读取）  
  方法：To address these challenges, we present SaSPA: Structure and Subject Preserving Augmentation.  
  结果：We propose SaSPA, a generative augmentation method specifically designed for FGVC. Our method generates diverse, class-consistent synthetic images through conditioning on edge maps and subject representation. SaSPA consistently outperforms both traditional and…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27c8b849acba6793f0b73f7ee7ea7397-Abstract-Conference.html)
- **Synthetic Data from Diffusion Models Improves ImageNet Classification**（全文PDF已读取）  
  方法：We show that large-scale text-to- image diffusion models can be ﬁne-tuned to produce class- conditional models with SOTA FID (1.76 at 256×256 reso- lution) and Inception Score (239 at 256 × 256).  
  结果：This paper asks to what extent generative data augmen- tation is effective with current diffusion models. We do so in the context of ImageNet classiﬁcation, a challenging do- main as it is extensively explored with highly tuned archi- tectures and training rec…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2304.08466)
- **Class-Mixed Diffusion Augmentation for Shortcut-Breaking in Continual Learning**（当前无本地PDF）  
  方法：全文中未提取到可靠方法摘要  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://neurips.cc/virtual/2026/loc/atlanta/poster/154098)
- **SynBalance: Harnessing Synthetic Data in Long-tailed Recognition**（全文PDF已读取）  
  方法：This paper presents a novel approach to tackling the long-tail problem by integrating strategically selected synthetic data, generated with state-of-the-art models like Stable Diffusion and DALL-E, to complement the orig…  
  结果：We present SynBalance, a comprehensive framework that leverages synthetic data to address the challenges of long- tailed distributions in model training. By investigating four critical properties of synthetic data—quantity, diversity, rec- ognizability, and do…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Jiang_SynBalance_Harnessing_Synthetic_Data_in_Long-tailed_Recognition_ICCVW_2025_paper.html)
- **Latent-based Diffusion Model for Long-tailed Recognition**（全文PDF已读取）  
  方法：We propose a new approach, the Latent-based Diffusion Model for Long-tailed Recognition (LDMLR), as a feature augmentation method to tackle the issue.  
  结果：This work proposes a novel framework, LDMLR, to ad- dress the challenge of long-tailed recognition. The LDMLR leverages the powerful generative capabilities of diffusion models for latent-level data augmentation, aiming to bal- ance long-tailed distributed dat…  
  开源：[https://github.com/AlvinHan123/LDMLR](https://github.com/AlvinHan123/LDMLR)；[论文来源](https://doi.org/10.1109/cvprw63382.2024.00270)
- **StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners**（全文PDF已读取）  
  方法：StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners Yonglong Tian1,∗ Lijie Fan1,2,∗ Phillip Isola2 Huiwen Chang1 Dilip Krishnan1 1Google Research, 2MIT CSAIL, ∗equal contribut…  
  结果：, Limitations and Broader Impact We have shown that solely synthetic data generated from state of the art text-to-image models can be used to train powerful visual representations. By harnessing the stochastic nature of Stable Diffusion in combination with a m…  
  开源：[https://github.com/google-research/syn-rep-learn](https://github.com/google-research/syn-rep-learn)；[论文来源](https://papers.neurips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html)
- **Fake It Till You Make It: Learning Transferable Representations From Synthetic ImageNet Clones**（全文PDF已读取）  
  方法：Provided only with the class names that have been used to build the dataset, we explore the ability of Stable Diffusion to generate synthetic clones of ImageNet and measure how useful these are for training classificatio…  
  结果：In this paper, we study to which extent ImageNet, ar- guably the most popular computer vision dataset, can be replaced by a dataset synthesized by a text-to-image gen- erator. Through an extensive study, we find that one can learn models that exhibit surprisin…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2023/html/Sariyildiz_Fake_It_Till_You_Make_It_Learning_Transferable_Representations_From_CVPR_2023_paper.html)
- **Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning**（全文PDF已读取）  
  方法：In this paper, we introduce a simple yet effective approach called Augmenting Discriminative Richness via Diffusions (AiR), toward learning a richer discriminating way to represent the class comprehensively and thus faci…  
  结果：In this paper, we introduce AiR, a novel auxiliary classifi- cation approach designed to leverage synthetic samples in building an auxiliary classifier. In this way, we can capture richer visual variations while preserving core semantics, ef- fectively bridgin…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Beyond_Words_Augmenting_Discriminative_Richness_via_Diffusions_in_Unsupervised_Prompt_CVPR_2025_paper.html)
- **ADuLTS: Appearance Descriptions under Long-Tailed Scenarios with Diverse Synthesized Images**（当前无本地PDF）  
  方法：全文中未提取到可靠方法摘要  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://www.sciencedirect.com/science/article/pii/S1077314226000512)
- **Synthetic Imagery Improves Ecological Classification When Real Data Are Scarce, Not Direct Substitutes**（当前无本地PDF）  
  方法：全文中未提取到可靠方法摘要  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.70105)
- **LoFT: LoRA-Fused Training Dataset Generation with Few-shot Guidance**（全文PDF已读取）  
  方法：In this paper, we introduce a novel dataset generation framework named LoFT, LoRA-Fused Training-data Generation with Few- shot Guidance.  
  结果：differ between fine-tuning and training from scratch. While the best method for training from scratch de- pends on the number of k-shot real images used for guidance, LoFT outperforms all baseline methods consistently across all k-shot settings when fine-tunin…  
  开源：[https://github.com/ExplainableML/LoFT.](https://github.com/ExplainableML/LoFT.)；[论文来源](https://arxiv.org/abs/2505.11703)

</details>

<details>
<summary>通用视觉/其他（11篇）</summary>

- **Advances in Diffusion Models for Image Data Augmentation: A Review of Methods, Models, Evaluation Metrics and Future Research Directions**（全文PDF已读取）  
  方法：全文中未提取到可靠方法摘要  
  结果：Image augmentation comprises a fundamental task in modern computer vision, since it allows the enhancement of training datasets with realistic synthetic samples, permitting the automatic context- and semantics-aware editing of given reference images, etc. Diff…  
  开源：正文未确认作者代码；[论文来源](https://doi.org/10.1007/s10462-025-11116-x)
- **Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models**（全文PDF已读取）  
  方法：Our extensive experiments show that by augmenting only 30%-40% of the data, our method boosts generalization by up to 2.8% in a variety of scenarios, including training ResNet, ViT, ConvNeXt, and Swin Transformer on CIFA…  
  结果：In this work, we show that synthetically augmenting part of the data that is not learned early in training outperforms augmenting the entire dataset. By analyzing a two-layer CNN, we prove that this method allows features to be learned at a more uniform speed …  
  开源：[https://github.com/lucidrains/vit-pytorch.](https://github.com/lucidrains/vit-pytorch.)；[论文来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html)
- **WILD-Diffusion: A WDRO Inspired Training Method for Diffusion Models under Limited Data**（全文PDF已读取）  
  方法：To address these limi- tations, we propose a novel method called WILD-Diffusion, which is inspired by Wasserstein Distributionally Robust Optimization (WDRO), an important and elegant mathematical formulation from robust…  
  结果：In this paper, we introduced WILD-Diffusion, a novel diffusion training framework based on WDRO. Our method dynamically expands the support of the training distribution, which miti- gates overfitting and improves generation quality under limited data. We propo…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9f709622912e87f8d84bffb097ae5503-Abstract-Conference.html)
- **Generative Data Augmentation via Diffusion Distillation, Adversarial Alignment, and Importance Reweighting (DAR-GDA)**（全文PDF已读取）  
  方法：We propose DAR-GDA, a three-stage augmentation pipeline that unites model Distillation, Adversarial alignment, and importance Reweighting that makes diffusion-quality augmentation both fast and optimized for improving do…  
  结果：. In this paper, we addressed two primary hindrances to the practical application of current diffusion models in GDA: suboptimal sampling efficiency and potential misalignment be- tween generated and target data distributions. We introduced DAR-GDA, a three-st…  
  开源：[https://github.com/ruyianry/gda-dar.](https://github.com/ruyianry/gda-dar.)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/62e5721247075dd097023d077d8e22f7-Abstract-Conference.html)
- **DIAGen: Diverse Image Augmentation with Generative Models**（全文PDF已读取）  
  方法：Finally, We introduce a weighting mechanism to mitigate the impact of poorly generated samples.  
  结果：We introduced DIAGen, an off-the-shelf image augmentation technique designed to increase semantic diversity in datasets with few labeled examples per class. DIAGen expands the DA-Fusion framework [47] by incorporating three key com- ponents: The first two modu…  
  开源：[https://github.com/visinf/DIAGen](https://github.com/visinf/DIAGen)；[论文来源](https://arxiv.org/abs/2408.14584)
- **RealSynz: Naturalizing Synthetic Images with Envelope Variation Statistics**（全文PDF已读取）  
  方法：Based on this finding, we present REALSYNZ, an independent module that corrects this distribution in the output of any generator without access to the model.  
  结果：AND FUTURE WORK Generated images distribute fine texture differently from natural images, and the moment ratio and envelope variation measure this difference, which energy and spectra miss. On FLUX.2 Klein, REALSYNZ removes 65% of the envelope variation gap, a…  
  开源：[https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)；[论文来源](https://openreview.net/challenge?redirect=%2Fforum%3Fid%3D80jkP3BMjb)
- **Scaling Laws of Synthetic Images for Model Training ... for Now**（全文PDF已读取）  
  方法：In this paper we study the scaling laws of synthetic images generated by state of the art text-to-image models, for the training of supervised models: image classi- fiers with label supervision, and CLIP with language su…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/google-research/syn-rep-learn](https://github.com/google-research/syn-rep-learn)；[论文来源](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Scaling_Laws_of_Synthetic_Images_for_Model_Training_..._for_CVPR_2024_paper.html)
- **Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data**（当前无本地PDF）  
  方法：全文中未提取到可靠方法摘要  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://neurips.cc/virtual/2026/loc/sydney/poster/149816)
- **FrozenDrive: Zero-Shot Text-Guided Driving Scene Generation and Data Augmentation with Parameter-Free Frozen Diffusion Model**（全文PDF已读取）  
  方法：We address these gaps with FrozenDrive, a controllable generative framework that preserves a pretrained diffusion model’s knowledge while achieving strong consistency.  
  结果：We propose FrozenDrive, a zero-shot text-guided multi-view driving scene gen- erator that steers a parameter-free frozen pretrained diffusion model. With our FrozenDrive 15 knowledge-preserving attention, FrozenDrive produces realistic, spatio-tempo- rally coh…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2606.20110)
- **Not Just Pretty Pictures: Toward Interventional Data Augmentation Using Text-to-Image Generators**（全文PDF已读取）  
  方法：We experiment across a diverse collection of benchmarks in single do- main generalization (SDG) and reducing reliance on spurious features (RRSF), ablating across key dimensions of T2I generation including interven- tion…  
  结果：In this work, we study the application of T2I generators to performing IDA in two settings, SDG and RRSF, finding they perform much better than traditional data augmentation techniques. We carry out a detailed investigation of how various components of the gen…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.mlr.press/v235/yuan24e.html)
- **Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning**（全文PDF已读取）  
  方法：In this paper, we propose a data-centric generative augmentation approach that leverages a diffusion model to enrich labeled data using both labeled and unlabeled samples.  
  结果：In this paper, we highlighted the potential of diffusion models for addressing class distribution mismatch in SSL. We introduced Discriminator-Weighted Diffusion (DWD), a semi-supervised training scheme that leverages a discriminator to identify OOD instances …  
  开源：[https://github.com/LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2024/file/777cc2af0ab984e6dc48d168ce7b754f-Paper-Conference.pdf)

</details>

<details>
<summary>综述/评测（3篇）</summary>

- **A Comprehensive Survey on Data Augmentation**（全文PDF已读取）  
  方法：This work was supported in part by the National Natural Science Foun- dation of China under Grant 62406306 and Grant 92470204 and in part by the National Key Research and Development Program of China under Grant 2024YFF0…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://people.cs.vt.edu/~clu/Publication/2026/Survey_on_Data_Augmentation-2026.pdf)
- **Quality Assessment and Model Training Impact of Synthetic Images in Medical Diagnostic Imaging: Systematic Review**（全文PDF已读取）  
  方法：We provide a quantitative cross-study synthesis of image-quality metrics and downstream model performance, comparing synthetic-only and synthetic-augmented training across different tasks and imaging modalities.  
  结果：Our systematic review provides a comprehensive quantita- tive assessment of synthetic-image quality evaluation and downstream model impact across diagnostic medical imaging studies. Unlike previous reviews that focused on specific diseases, modalities, or gene…  
  开源：[https://github.com/DS3-SisCat/2507-synthetic-images-sysrev/](https://github.com/DS3-SisCat/2507-synthetic-images-sysrev/)、[https://github.com/cebudding/MedNeurips_](https://github.com/cebudding/MedNeurips_)；[论文来源](https://ai.jmir.org/2026/1/e81472/)
- **When More Data Hurts: A Critical Survey of Failure Modes in Generative Data Augmentation for Image Classification**（全文PDF已读取）  
  方法：Drawing on evidence from over 30 papers across computer vision, medical imaging, and fine-grained recog- nition, we provide the first systematic taxonomy of failure modes in generative data augmentation, that is, conditi…  
  结果：Generative data augmentation is a powerful tool, but it is not a universally safe intervention. This survey provides the first systematic taxonomy of failure modes in GDA, drawing on evidence from over 30 papers to identify seven distinct mechanisms by which s…  
  开源：正文未确认作者代码；[论文来源](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6448800)

</details>

<details>
<summary>分割/稠密预测（22篇）</summary>

- **MatchMask: Mask-Centric Generative Data Augmentation for Label-Scarce Semantic Segmentation**（全文PDF已读取）  
  方法：In this paper, we propose MatchMask, a novel mask-centric generative data augmentation approach tailored for label- scarce semantic segmentation.  
  结果：This paper introduces MatchMask, a mask-centric genera- tive data augmentation framework tailored for label-scarce semantic segmentation. It incorporates three key contribu- tions: 1) Gradient Probe Method to systematically iden- tify critical layers for mask-…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_MatchMask_Mask-Centric_Generative_Data_Augmentation_for_Label-Scarce_Semantic_Segmentation_CVPR_2026_paper.html)
- **S3OD: Towards Generalizable Salient Object Detection with Synthetic Data**（全文PDF已读取）  
  方法：We present a method that dramatically improves generalization through large-scale synthetic data generation and ambiguity-aware architecture.  
  结果：We demonstrate that combining features from generative and discriminative models: diffusion trans- former feature maps, concept attention maps, and DINO-v3 features enables effective synthetic data generation for salient object detection. Our iterative generat…  
  开源：[https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)、[https://github.com/black-forest-labs/flux.](https://github.com/black-forest-labs/flux.)；[论文来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ead7401f3a05d5285f49a915cfac2f2a-Abstract-Conference.html)
- **Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation**（全文PDF已读取）  
  方法：We introduce Sketch2CT, a multimodal diffusion framework for structure- aware 3D medical volume generation, jointly guided by a user-provided 2D sketch and a textual description that captures 3D geometric semantics.  
  结果：and Future Work We have presented Sketch2CT, a multimodal diffusion framework that generates structure-aware 3D medical vol- umes from sketches and textual descriptions. The frame- work comprises a segmentation generator that reconstructs anatomical structures…  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/An_Sketch2CT_Multimodal_Diffusion_for_Structure-Aware_3D_Medical_Volume_Generation_CVPR_2026_paper.html)
- **Sample-Efficient Multi-Round Generative Data Augmentation for Long-Tail Instance Segmentation (MRCA)**（全文PDF已读取）  
  方法：To address this inefficiency, we propose a collaborative approach that incorporates feedback from an instance segmentation model to guide the augmentation process.  
  结果：The proposed multi-round collaborative augmentation (MRCA) framework enables generating objects that are continuously more informative to the instance segmentation model with a considerably small amount of data generations (e.g., 6%). The instance segmentation…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/69f59f96c3aae0ba9f463e7903715e1e-Abstract-Conference.html)
- **Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models**（全文PDF已读取）  
  方法：In contrast to these isolated and thus sub-optimal efforts, we introduce an integrated, versatile, diffusion-based framework, Diff-2-in-1, that can simultaneously handle both multi-modal data generation and dense visual …  
  结果：In this paper, we bridge generative and discriminative learning by proposing an integrated diffusion- based framework Diff-2-in-1. It enhances discriminative learning through the generative process by creating diverse while faithful data, and gets the discrimi…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1ec299a5229034141e58aeded0d0b9de-Abstract-Conference.html)
- **MedDiff-FT: Data-Efficient Diffusion Model Fine-Tuning with Structural Guidance for Controllable Medical Image Synthesis**（全文PDF已读取）  
  方法：To address these challenges, we present MedDiff- FT, a controllable medical image generation method that fine-tunes a diffusion foundation model to produce medical images with structural dependency and domain specificity…  
  结果：This paper proposes a resource-efficient controllable generation paradigm for diffusion models using limited data. By fine-tuning Stable Diffusion with min- imal training data, our method produces high-quality image-mask pairs vali- dated through downstream se…  
  开源：[https://github.com/JianhaoXie1/MedDiff-FT.](https://github.com/JianhaoXie1/MedDiff-FT.)；[论文来源](https://doi.org/10.1007/978-3-032-04965-0_29)
- **DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models**（全文PDF已读取）  
  方法：In this paper, we show that it is possible to automatically obtain accurate semantic masks of synthetic images generated by the Off-the-shelf Stable Diffusion model, which uses only text-image pairs during training.  
  结果：A new insight is presented in this paper, demonstrat- ing that the accurate semantic mask of generative images can be automatically obtained through the use of a text- driven diffusion model. To achieve this goal, we present DiffuMask, an automatic procedure t…  
  开源：正文未确认作者代码；[论文来源](https://ieeexplore.ieee.org/document/10376904/)
- **DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models**（全文PDF已读取）  
  方法：In this paper, we present DatasetDM, a generic dataset generation model that can produce diverse synthetic images and the corresponding high-quality perception annotations (e.g., segmenta- tion masks, and depth).  
  结果：In this study, we investigate using a perception decoder to parse the latent space of an advanced diffusion model, extending the text-to-image task to a new paradigm: text-guided data generation. Training the decoder requires less than 1% of existing labeled i…  
  开源：[https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html)
- **Improving Diffusion-Based Augmentation for Data-Scarce Image Classification via Mask-Guided Reference Attention and Repulsive Noise Optimization**（全文PDF已读取）  
  方法：To overcome these limitations, we propose a sampling-side framework of two complementary components that simultaneously en- hance fine details and diversity.  
  结果：We introduced a sampling-time diffusion augmentation framework combining MGR Attention and RNO for data- scarce image classification. MGR Attention updates only foreground value tokens with reference features via zero- shot segmentation, strengthening fine-gra…  
  开源：[https://github.com/hvcl/MGRAttn-RNO](https://github.com/hvcl/MGRAttn-RNO)；[论文来源](https://arxiv.org/abs/2611.04242)
- **Balancing Representations by Identifying and Generating Under-represented Data (BRIDGE)**（全文PDF已读取）  
  方法：We introduce BRIDGE (Balancing Representations by Identifying and Generating Under-represented Data), a label- free feedback loop around the source set.  
  结果：In this paper we introduce BRIDGE, which lets a self-supervised learner fill the sparse regions of its own representation and improves transfer from every long-tailed source we tested. Web-scale pretraining corpora are long-tailed in the same way as our source…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2611.02952)
- **Free-Lunch Augmentation by Revisiting Diffusion-Based Data Generation for Cross-Domain Few-Shot Object Detection**（全文PDF已读取）  
  方法：To ad- dress this problem, we revisit a natural yet underexplored approach in CDFSOD: data augmentation, by directly synthesizing data through dif- fusion models to supplement limited training samples.  
  结果：We revisit a neglected method for the CDFSOD task: diffusion-based data aug- mentation, finding that domain gaps make it difficult for current diffusion-based methods to synthesize useful samples for supplementing the scarce training data. We then analyze the …  
  开源：[https://github.com/zzzzj311-droid/Free-Lunch-SITN.](https://github.com/zzzzj311-droid/Free-Lunch-SITN.)；[论文来源](https://arxiv.org/abs/2608.04394)
- **Post-Generation Curation of Synthetic Images via Homogeneous-Heterogeneous Splitting**（全文PDF已读取）  
  方法：We study a complementary question: given a fixed pool of generated images, can downstream utility be improved purely by selecting an informative subset?  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2607.02637)
- **Learning Vision from Models Rivals Learning Vision from Data (SynCLR)**（全文PDF已读取）  
  方法：Learning Vision from Models Rivals Learning Vision from Data Yonglong Tian1,† Lijie Fan2,†,* Kaifeng Chen1 Dina Katabi2 Dilip Krishnan1 Phillip Isola2 1Google Research, 2MIT CSAIL, †equal contribution Github Repo: https:…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/google-research/syn-rep-learn](https://github.com/google-research/syn-rep-learn)；[论文来源](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Learning_Vision_from_Models_Rivals_Learning_Vision_from_Data_CVPR_2024_paper.html)
- **MRGen: Segmentation Data Engine For Underrepresented MRI Modalities**（全文PDF已读取）  
  方法：This paper investigates leveraging generative models to syn- thesize data, for training segmentation models for under- represented modalities, particularly on annotation-scarce MRI.  
  结果：This paper explores generative models for controllable MRI generation, particularly to facilitate training segmentation models for underrepresented modalities lacking mask an- notations. To support this, we curate a large-scale radi- ology image-text dataset, …  
  开源：正文未确认作者代码；[论文来源](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MRGen_Segmentation_Data_Engine_For_Underrepresented_MRI_Modalities_ICCV_2025_paper.html)
- **Learning Diverse and Realistic Polyp Data via Style-Fused Diffusion Models**（全文PDF已读取）  
  方法：To address these issues, we propose SFD-Polyp, a style-fused conditional diffusion framework that conditions generation on ground-truth masks and injects appearance cues via a Style Integration Module (SIM), enhancing te…  
  结果：This work proposes the style-fused diffusion framework SFD-Polyp, which en- hances the style realism of generated images via SIM and expands mask mor- phological diversity through MPMS, enriching the conditional space for data generation. Experiments show that…  
  开源：[https://github.com/zhaiying0925/SFD-](https://github.com/zhaiying0925/SFD-)；[论文来源](https://papers.miccai.org/miccai-2026/0557-Paper4818.html)
- **Dual-Stream Diffusion with Structure-Aware Fusion for Joint Synthesis of Coherent Medical Image-Mask Pairs**（全文PDF已读取）  
  方法：We propose Dual-Stream Diffu- sion (DSD), a joint generative framework that models paired data with two decoupled denoising streams: a structure stream for mask predic- tion and an appearance stream for image synthesis, …  
  结果：We proposed Dual-Stream Diffusion (DSD), a pixel-space joint diffusion frame- work for generating aligned medical image-mask pairs. By decoupling structure and appearance into two denoising streams and introducing time-gated fusion and structure-aware heads mo…  
  开源：正文未确认作者代码；[论文来源](https://doi.org/10.1007/978-3-032-38189-7_18)
- **Learning to Segment Burns with Limited Labels: A Dual-Path Cascaded Network with Uncertainty-Guided Teacher Diffusion Exposure**（全文PDF已读取）  
  方法：We propose UADA-MSPCNet, a semi- supervised segmentation framework that improves annotation efficiency and boundary delineation.  
  结果：We presented a semi-supervised framework, UADA-MSPCNet, that improves burn segmentation by jointly optimizing representation learning and pseudo- label calibration. Experimental results demonstrate enhanced boundary delin- eation and annotation efficiency, wit…  
  开源：[https://github.com/VEDAs-Lab/EBIS.](https://github.com/VEDAs-Lab/EBIS.)；[论文来源](https://papers.miccai.org/miccai-2026/0573-Paper5539.html)
- **Polyp-LDM: Accurate Boundary Alignment and Realism Enhancement for Colonoscopic Polyp Image-Mask Pair Generation**（全文PDF已读取）  
  方法：This paper proposes Polyp- LDM that encodes polyps and masks into the same latent space via a unified variational autoencoder (VAE) to align their boundaries.  
  结果：This paper proposes Polyp-LDM for paired polyp image-mask generation in colonoscopy. Recently, generative models for complex and critical endoscopic scenes have drawn significant attention [16, 8]. Our method achieves precise boundary alignment by unifying the…  
  开源：[https://github.com/16rq/Polyp-LDM.](https://github.com/16rq/Polyp-LDM.)；[论文来源](https://papers.miccai.org/miccai-2025/0037-Paper2379.html)
- **DiDGen: Diffusion-Based Dual-Task Synthesis for Dermoscopic Data Generation**（全文PDF已读取）  
  方法：To address this limitation, we propose DiDGen, a novel method that employs Diffusion models (DMs) for Dermoscopic image Generation and lesion-mask pair synthesis.  
  结果：We introduce DiDGen, a novel diffusion-based model for dermoscopic genera- tion. Specifically, we propose DermPrompt, a structured text prompt paradigm DiDGen 9 containing rich clinical details annotated by LLMs. Additionally, our approach generates accurate l…  
  开源：[https://github.com/junjie-shentu/DiDGen.](https://github.com/junjie-shentu/DiDGen.)；[论文来源](https://doi.org/10.1007/978-3-032-05141-7_8)
- **DiffAtlas: GenAI-fying Atlas Segmentation via Image-Mask Diffusion**（全文PDF已读取）  
  方法：In contrast, we introduce DiffAt- las, a novel generative framework that models both images and masks through diffusion during training, effectively “GenAI-fying” atlas-based segmentation.  
  结果：In conclusion, we present DiffAtlas, which jointly models image-mask pairs dur- ing training and guides the model to generate target image-mask pairs, ulti- mately producing the corresponding masks during testing. This approach alle- viates the limitations of …  
  开源：[https://github.com/M3DV/DiffAtlas.](https://github.com/M3DV/DiffAtlas.)、[https://github.com/google-deepmind/surface-](https://github.com/google-deepmind/surface-)；[论文来源](https://papers.miccai.org/miccai-2025/0231-Paper0228.html)
- **Triple-Prompt Controllable Diffusion for Universal Data Augmentation in Medical Image Segmentation**（当前无本地PDF）  
  方法：全文中未提取到可靠方法摘要  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://journals.sagepub.com/doi/10.3233/FAIA250876)
- **Diffusion-Based Data Augmentation for Medical Image Segmentation**（全文PDF已读取）  
  方法：We propose DiffAug a novel framework that combines text- guided diffusion-based generation with automatic segmen- tation validation to address this challenge.  
  结果：We presented a framework that addresses data scarcity in medical image segmentation by combining text-guided diffusion-based generation with automatic quality valida- tion. The framework reduces false negative rates for chal- lenging cases like small polyps an…  
  开源：正文未确认作者代码；[论文来源](https://doi.org/10.1109/iccvw69036.2025.00143)

</details>

<details>
<summary>数据集蒸馏/数据集构建（8篇）</summary>

- **IMS3: Breaking Distributional Aggregation in Diffusion-Based Dataset Distillation**（全文PDF已读取）  
  方法：Our method covers a broader area of the feature space, indicating improved distribu- tional coverage.  
  结果：We propose ImS3, a simple yet effective two-stage frame- work for large-scale dataset distillation, addressing two crit- ical issues in diffusion-based approaches: insufficient dis- tribution coverage and weak discriminative diversity. Our method achieves stat…  
  开源：[https://github.com/Westlake-AGI-Lab/IMS3.](https://github.com/Westlake-AGI-Lab/IMS3.)；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_IMS3_Breaking_Distributional_Aggregation_in_Diffusion-Based_Dataset_Distillation_CVPR_2026_paper.html)
- **EVLF: Early Vision-Language Fusion for Generative Dataset Distillation**（全文PDF已读取）  
  方法：To solve this problem, we introduce an Early Vision-Language Fusion (EVLF) method that aligns tex- tual and visual embeddings at the transition between the encoder and the generative backbone.  
  结果：We introduced Early Vision-Language Fusion (EVLF), a plug-and-play method that integrates textual semantics into the visual latent space before denoising through a lightweight cross-attention mechanism. By grounding se- mantic cues at the encoder stage, EVLF m…  
  开源：[https://github.com/Guang000/](https://github.com/Guang000/)；[论文来源](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_EVLF_Early_Vision-Language_Fusion_for_Generative_Dataset_Distillation_CVPR_2026_paper.html)
- **Expanding Small-Scale Datasets with Guided Imagination (GIF)**（全文PDF已读取）  
  方法：To this end, we present a Guided Imagination Framework (GIF) that leverages cutting-edge generative models like DALL-E2 and Stable Diffusion (SD) to "imagine" and create informative new data from the input seed data.  
  结果：This paper has explored a novel task, dataset expansion, towards resolving the data scarcity issue in DNN training. Inspired by human learning with imagination, we presented a novel guided imagination framework for dataset expansion. Promising results on small…  
  开源：[https://github.com/Vanint/DatasetExpansion.](https://github.com/Vanint/DatasetExpansion.)；[论文来源](https://papers.neurips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html)
- **Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation**（全文PDF已读取）  
  方法：We propose a prompt- controlled diffusion augmentation framework that generates paired label-image samples with explicit control over semantic composition and domain, enabling targeted enrichment of under- represented cl…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2602.04749)
- **D3S2: Diffusion-Guided Dataset Distillation for Semantic Segmentation**（全文PDF已读取）  
  方法：To address these challenges, we propose D3S2, a Diffusion-guided Dataset Distillation framework for Semantic Segmentation.  
  结果：This paper presents D3S2, a diffusion-guided dataset distillation framework for semantic segmentation. It addresses the unique challenges of dense prediction via a two-stage design: (i) a class-balanced greedy selection strategy that constructs a compact and r…  
  开源：[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)、[https://github.com/zwj084/D3S2.](https://github.com/zwj084/D3S2.)；[论文来源](https://arxiv.org/abs/2605.25022)
- **S2-RL: Sample-Set Dual Reinforcement Learning for Generative Semantic Segmentation Dataset Distillation**（全文PDF已读取）  
  方法：To address both bottlenecks simultaneously, we propose a novel Sample-Set Re- inforcement Learning (S2-RL) framework for generative semantic segmentation dataset distillation (SSDD).  
  结果：and Future Work We propose S2-RL, a generative framework for semantic segmentation dataset distillation that ﬁne- tunes Stable Diffusion via GRPO with a sample-set dual reward, integrating sample-level ﬁdelity and set-level coverage to bypass the computational…  
  开源：[https://github.com/00why00/S2RL](https://github.com/00why00/S2RL)、[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)；[论文来源](https://arxiv.org/abs/2610.35491)
- **DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery**（全文PDF已读取）  
  方法：To address this, we propose a novel dual-stage distillation framework called DIVER, which leverages the pre-trained diffusion model to dive deeper into DIstilled data Via Expressive semantic Recovery, an entire process o…  
  结果：In this paper, we propose a novel task of “Diving into Dis- tilled Data” for the first time, where we enhance remarkable cross-architecture generalization by recovering the expres- sive semantics suppressed by specific patterns in the distilled dataset. Our me…  
  开源：正文未确认作者代码；[论文来源](https://proceedings.mlr.press/v306/xia26g.html)
- **Efficient Dataset Distillation for Pre-Trained Self-Supervised Models via Statistical Flow Matching**（全文PDF已读取）  
  方法：To address this, we introduce Statistical Flow Matching, an optimal, stable, and efficient supervised learning framework that optimizes synthetic images by aligning global statistical flows in the original data.  
  结果：In this work, we are the first to propose the concept of “flow” in DD, from which we derive a novel paradigm termed Statistical Flow Matching that encourages the synthetic images to efficiently learn the statistical flow of the original dataset through the pre…  
  开源：[https://github.com/einsteinxia/SFM.](https://github.com/einsteinxia/SFM.)；[论文来源](https://arxiv.org/abs/2602.05391)

</details>

<details>
<summary>医学影像（5篇）</summary>

- **The Learnability Gap in Medical Latent Diffusion**（全文PDF已读取）  
  方法：Across five autoencoder families and four medical benchmarks spanning chest radiography, der- matoscopy, computed tomography, and echocardiography, we show that this gap persists regardless of architecture, initializatio…  
  结果：We introduced the learnability gap: a systematic discrepancy between the dis- criminative information preserved in autoencoder latent spaces and the ability of downstream models to access it. Through a comprehensive exploration spanning five autoencoder famili…  
  开源：[https://github.com/MischaD/LearnabilityGap.git.](https://github.com/MischaD/LearnabilityGap.git.)；[论文来源](https://papers.miccai.org/miccai-2026/1053-Paper3049.html)
- **Generate What Matters: Steering Diffusion Models for Targeted Data Generation to Improve Classification (C2I)**（全文PDF已读取）  
  方法：Building on this insight, we fine-tune diffusion models with reinforcement learning using a C2I-based reward to steer generation to- ward class-informative samples.  
  结果：We investigated how to fine-tune diffusion models to generate more effective training samples for few-shot classification. Our analysis showed that the most useful samples exhibit a large influence gap between two classes: their gradients are aligned with vali…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2607.12464)
- **Synthetic Data Generation for Long-Tail Medical Image Classification: A Case Study in Skin Lesions**（全文PDF已读取）  
  方法：To address these challenges, we introduce a diffusion-model-driven synthetic data augmentation pipeline tailored for med- ical long-tailed classification.  
  结果：In this paper, we proposed a novel synthetic data gener- ation approach to address the challenges of LTMIC, us- ing skin lesion classification as a primary case study. Our method integrates a novel diffusion model with out-of- distribution (OOD) filtering to e…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2605.03221)
- **Semi-Supervised Domain Adaptation with Latent Diffusion for Pathology Image Classification**（全文PDF已读取）  
  方法：In this work, we propose a semi-supervised domain adaptation (SSDA) framework that utilizes a latent diffusion model trained on unlabeled data from both the source and target domains to generate morphology-preserving and…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/hsu-lab/ldm-pathology-ssda.](https://github.com/hsu-lab/ldm-pathology-ssda.)；[论文来源](https://doi.org/10.1109/JBHI.2026.3727978)
- **TRCGL-Net: A Long-Tailed Multi-Label Chest X-Ray Classification Framework with Generative Data Augmentation and Label Co-Occurrence Modeling**（全文PDF已读取）  
  方法：To address these key challenges, we propose a TRCGL-Net framework for long-tailed multi-label classification of chest X-rays (CXR).  
  结果：To address the challenges of long-tailed multi-label chest X-ray classification, including severe sample imbalance in tail classes, subtle lesion characteristics, and insufficient exploitation of label co-occurrence information, this study proposes the TRCGL-N…  
  开源：[https://github.com/November-1113/TRCG](https://github.com/November-1113/TRCG)；[论文来源](https://arxiv.org/abs/2607.00975)

</details>

<details>
<summary>检测（5篇）</summary>

- **Improving Long-Tail Detection via Layout-Instruction Conditioned Diffusion Synthesis (Tail-SD)**（全文PDF已读取）  
  方法：In this paper, we propose Tail-SD, a lay- out–instruction conditioned stable diffusion framework that sup- ports text, layouts, or their hybrid, enabling controllable synthesis from boxes, masks, and/or prompts to better…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：[https://github.com/ChenyunWu/PhraseCutDataset](https://github.com/ChenyunWu/PhraseCutDataset)、[https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)、[https://github.com/yoctta/XPaste/issues/2](https://github.com/yoctta/XPaste/issues/2)；[论文来源](https://arxiv.org/abs/2611.07553)
- **Few-Shot Synthetic Data Generation with Diffusion Models for Downstream Vision Tasks**（全文PDF已读取）  
  方法：We propose a lightweight synthetic data augmentation pipeline that fine-tunes a LoRA adapter on as few as 20–50 real images of a rare class and uses a pretrained diffusion model to generate synthetic sam- ples for traini…  
  结果：当前证据未提取到可核对的结果，需回到全文或补充材料  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2605.11898)
- **Usefulness of Quantile-Aware Diffusion Modeling for Highly Imbalanced Tabular Data**（全文PDF已读取）  
  方法：We propose a novel approach, namely, Quantile-TabDDPM, based on a quantile-regularized denoising objective that combines the standard quadratic error loss with a quantile loss term to explicitly capture rare events while…  
  结果：In this study, we proposed Quantile-TabDDPM, a diffusion-based generative framework for numerical tabular data that incorporates quantile regularization into the denoising objective. Experimental results on the ECC dataset demon- strate improved fraud detectio…  
  开源：正文未确认作者代码；[论文来源](https://arxiv.org/abs/2610.05825)
- **DiffAug: A Diffuse-and-Denoise Augmentation for Training Robust Classifiers**（全文PDF已读取）  
  方法：Abstract We introduce DiffAug, a simple and efficient diffusion-based augmentation tech- nique to train image classifiers for the crucial yet challenging goal of improved classifier robustness.  
  结果：In this work, we introduce DiffAug to train robust classifiers with one-step diffusion denoised examples. The simplicity and computational efficiency of DiffAug enables us to also extend other data augmentation techniques, where we find that DiffAug confers ad…  
  开源：[https://github.com/oore-lab/diffaug](https://github.com/oore-lab/diffaug)；[论文来源](https://proceedings.neurips.cc/paper_files/paper/2024/file/24e8b46430df965674221665816a4964-Paper-Conference.pdf)
- **Diffusion-Based User-Guided Data Augmentation for Coronary Stenosis Detection**（全文PDF已读取）  
  方法：In this study, we propose a novel data augmentation approach that uses an in- painting method based on a diffusion model to generate realistic lesions, allowing user-guided control of severity.  
  结果：In this study, we propose a novel approach to improve lesion detection and sever- ity classification, integrating synthetic data generated via a diffusion model to vessel narrowing simulations into the training process. User-guided modifica- tions to vessel ma…  
  开源：[https://github.com/medipixel/DiGDA.](https://github.com/medipixel/DiGDA.)；[论文来源](https://papers.miccai.org/miccai-2025/0238-Paper3613.html)

</details>

## 阅读边界

本页使用的主体证据来自 Zotero 本地 PDF 和条目中的论文来源链接。5 篇没有本地 PDF 的记录已经明确标注为“当前无本地 PDF”，它们在当前版本中作为待补全文献处理。正式接收、预印本、workshop 和 OpenReview 在审状态也按 Zotero 条目和来源页面分别保留。
