---
title: 扩散模型合成图像用于数据增强的研究脉络与方法比较
date: 2026-11-18
summary: 扩散模型数据增广正从“生成更多图像”转向“生成对下游任务有用的图像”。本文从 Zotero 候选库 169 篇中纳入 83 条直接相关证据（81 篇已读全文），按下游任务、数据条件和方法阶段三轴梳理 2023—2026 年的研究脉络，并比较代表性方法。
tags: [扩散模型, 合成数据, 数据增强, 计算机视觉, 综述]
tldr: 扩散模型数据增广正从“生成更多图像”转向“生成对特定模型、特定缺口有用的图像”；本文用任务、数据条件、方法阶段三条轴线梳理 83 篇文献，并比较这一转向中的代表方法。
authors: [MaybeRichard]
---

## 摘要

扩散模型让“用生成图像扩充训练集”从个别数据集上的技巧，变成了覆盖分类、分割、检测和表征学习的通用手段。本文基于 Zotero 中 83 条与图像合成增广直接相关的文献（其中 81 篇读取了全文），梳理 2023—2026 年的研究脉络，并比较代表性方法。文献按三条可交叉的轴线组织：主任务决定评价指标，数据条件决定生成要补的缺口，方法阶段决定生成器如何接入训练流程。

研究重心经历了三次转移。2023 年的工作证明，合成图像本身可以作为训练数据。2024 年转向在少样本条件下兼顾保真度与多样性。2025—2026 年进入任务感知阶段：让下游模型的反馈引导生成，或在生成后筛选真正有用的样本，并扩展到长尾识别和医学稠密预测。

比较结果显示，“生成什么”比“生成多少”更重要。只增强训练早期未学会的 30%–40% 样本，或只用约 6% 的生成预算，就能取得稳定收益。但各论文在真实数据量、生成器状态、筛选预算和混合方式上差异很大，指标不宜直接横向比较。文章最后归纳了六类风险，并提出下一步的关键：可复现的任务效用评测、可靠的样本筛选、结构一致的标注生成，以及跨域外部验证。

**关键词**：扩散模型；合成数据；数据增强；长尾识别；少样本学习；医学图像分割；任务反馈生成；样本筛选

## 1. 范围和读法

Zotero 中「合成图像用于数据增强」当前子树有 169 篇候选记录（已移除 1 条重复条目），Diffusion Model 子树有 219 篇更广义的扩散模型记录。本综述只纳入 83 条与图像合成增广直接相关的证据，其中 81 篇已读取本地 PDF，2 篇暂时只有题目、摘要或出版页面。参考文献列表共 89 条，包含 6 条背景文献和 83 条主体记录。


本版按此前质量筛选规则将 Sensors 2026 的 Propose-and-Select，以及此前明确排除的 ADuLTS、Synthetic Imagery 和 Triple-Prompt，均移出核心综述统计，条目仍可在 Zotero 中保留。

主任务使用单一归属，避免把同一篇论文重复计入任务总数。数据条件和方法阶段是交叉索引，因此它们的统计数不能相加为 83。WILD-Diffusion 和 Sketch2CT 保留在综述证据中，但分别归入生成模型基础与训练、医学与三维领域，不强行放入下游增广任务比较。

![83 条综述证据的三轴分类](three-axis-overview.svg)

<small>任务统计按主任务计算；数据条件和方法阶段允许交叉归属。图中不代表 Zotero 全库 169 篇候选记录。</small>

## 2. 发展脉络

### 2.1 2023：从生成图像走向生成训练数据

<a href="#ref-2240">Azizi 等人的工作</a>把类条件扩散模型用于 ImageNet 数据生成，证明合成图像能够支持分类器训练。<a href="#ref-2162">GIF</a> 将问题表述为 dataset expansion，用少量真实图像引导 DALL-E 2 或 Stable Diffusion 生成新的带标签样本，同时指出 synthetic-only 仍然不如同规模真实数据。

<a href="#ref-364">DiffuMask</a> 和 <a href="#ref-2166">DatasetDM</a> 将数据增广对象从图像扩展到图像与感知标注。DiffuMask 观察文本条件扩散中的 cross-attention，利用它自动得到像素级语义掩码。DatasetDM 在扩散模型的 latent 上加入感知解码器，可以生成图像、语义分割、实例分割和深度等配对数据。论文报告的代表性结果是 VOC 语义分割提升 13.3 个 mIoU，COCO 实例分割提升 12.1 个 AP。<a href="#ref-2258">StableRep</a> 则把合成数据用于视觉表示学习，将同一个文本提示生成的多个图像作为正样本进行多正例对比学习。

### 2.2 2024：控制保真度、多样性和少样本适配

<a href="#ref-2158">DA-Fusion</a> 从真实图像编辑起点开展数据增广，起点位于真实图像而非纯噪声，使用文本变化描述控制颜色、材质和场景变化。这样可以更好地保留原图结构，也能处理预训练模型没有直接见过的概念。

<a href="#ref-123">DiffuseMix</a> 使用条件提示重塑输入图像，再通过拼接和分形混合构造数据增广结果。<a href="#ref-2154">Diff-Mix</a> 进一步利用跨类别图像插值增加变化，明确处理 faithfulness 与 diversity 的平衡。<a href="#ref-2150">DataDream</a> 用少量真实样本对文本编码器和 U-Net 做 LoRA 微调，分别提供全数据集和逐类别的适配方式。它说明在 few-shot 条件下，轻量适配可以明显缩小生成分布和目标分布之间的差距。

这一阶段还出现了两条并行路线。<a href="#ref-2238">DiffAug</a> 研究 diffuse-and-denoise 数据增广对鲁棒分类的作用，<a href="#ref-2260">SynCLR</a> 和 StableRep 把重点放在合成数据训练视觉表示。扩散模型开始从下游训练集的“附加样本生成器”变成预训练数据源。

### 2.3 2025—2026：任务感知、长尾和医学场景

<a href="#ref-2118">Diff-II</a> 把扩散反演后的潜变量放在一个插值圆上，通过两阶段去噪同时提高生成样本的保真度和多样性。<a href="#ref-2120">MRCA</a> 让实例分割器反过来指导扩散模型，多轮生成模型不确定的对象，并按类别预算把对象放回真实图像。<a href="#ref-2098">DBG</a> 则围绕决策边界生成样本，并用分类器清洗有害样本。<a href="#ref-2122">CORAL</a> 发现长尾类别在 U-Net latent 中可能与头部类别重叠，于是用对比正则化解耦类别表示。<a href="#ref-2130">DAR-GDA</a> 通过蒸馏、对抗对齐和重要性重加权同时解决采样效率和分布偏移。

医疗和稠密预测工作更加关注结构一致性。<a href="#ref-2054">MatchMask</a> 从少量语义掩码生成图像—掩码对，用 Gradient Probe 找到关键层，再用轻量适配器和相对过滤抑制错误区域。<a href="#ref-2059">MRGen</a> 以文本和掩码共同控制 MRI 生成，服务于缺少标注的模态。<a href="#ref-2039">Dual-Stream Diffusion</a> 将结构流和外观流分开去噪，再用时间门控融合。<a href="#ref-2274">Polyp-LDM</a>、<a href="#ref-2035">DiDGen</a>、<a href="#ref-2276">DiffAtlas</a>、医学 DiffAug 和 MRGen 都把图像质量和下游分割效果同时作为评价目标。

2026 年的一个重要变化是评测开始系统化。<a href="#ref-2090">UniDiffDA</a> 将方法拆成模型微调、样本生成和样本利用三个阶段，在通用、细粒度、长尾、跨域和医学分类上重新实现多个方法。结论呈现出明显的任务依赖性，三阶段组合与目标任务共同决定结果。这个视角比单一榜单更适合解释论文之间为何会得到不同结论。

## 3. 按下游任务

本综述的主任务归属采用单一主类，数据条件和方法阶段作为交叉索引。83 条证据中，分类 39 条、目标检测 2 条、分割与稠密预测 24 条、表征学习 2 条、数据集蒸馏 8 条、综述与评测 6 条，以及 WILD-Diffusion、Sketch2CT 两条生成器与领域基础记录。

### 3.1 分类

分类论文覆盖从冻结生成器提示到领域适配、任务反馈和生成后筛选。Synthetic Data from Diffusion Models、GIF、StableRep、SynCLR 和 Fake It Till You Make It 说明，类别提示可以快速扩大覆盖，但细粒度类别容易受到背景捷径和提示偏差影响。DA-Fusion、DiffuseMix、Diff-Mix、DataDream、DIAGen、AugGen、BOB、ChimeraLoRA、LoFT 和 1S-DAug 分别从真实图像编辑、跨类别混合、LoRA 适配、上下文建模和少样本控制等方面提高类内变化与分布贴合度。<a href="#ref-2240">Synthetic Data from Diffusion Models</a>、<a href="#ref-2162">GIF</a>、<a href="#ref-2258">StableRep</a>、<a href="#ref-2260">SynCLR</a>、<a href="#ref-2262">Fake It Till You Make It</a>、<a href="#ref-2158">DA-Fusion</a>、<a href="#ref-123">DiffuseMix</a>、<a href="#ref-2154">Diff-Mix</a>、<a href="#ref-2150">DataDream</a>、<a href="#ref-2174">DIAGen</a>、<a href="#ref-2102">BOB</a>、<a href="#ref-2100">ChimeraLoRA</a>、<a href="#ref-2281">LoFT</a>、<a href="#ref-2210">1S-DAug</a>

分类任务中的改进通常来自更合适的样本，而不是简单增加合成数量。TADA 只扩增训练早期没有学到的样本，C2I 根据分类梯度构造目标，Post-Generation Curation 和 BRIDGE 则在生成后寻找更有用或欠表示的样本。<a href="#ref-116">TADA</a>、<a href="#ref-2190">C2I</a>、<a href="#ref-2250">Post-Generation Curation</a>、<a href="#ref-2182">BRIDGE</a>

### 3.2 目标检测

检测方向的证据数量较少，但方法阶段更加清晰。S3OD 面向显著目标检测的跨域泛化，Free-Lunch 处理跨域少样本目标检测，Tail-SD 用布局指令控制长尾目标生成，冠脉狭窄检测工作则把用户反馈直接接入增广过程。检测任务不能只看生成图像的视觉质量，还要检查目标位置、尺度、遮挡和框标注的一致性。<a href="#ref-2104">S3OD</a>、<a href="#ref-2244">Free-Lunch</a>、<a href="#ref-2186">Tail-SD</a>、<a href="#ref-2279">Diffusion-Based User-Guided Data Augmentation for Coronary Stenosis Detection</a>

### 3.3 分割与稠密预测

分割与稠密预测要求图像和 mask、layout、depth 等标注保持同步。DiffuMask 从 cross-attention 中提取像素级掩码，DatasetDM 用感知解码器生成图像和多种标注，MatchMask、MRGen、Dual-Stream Diffusion、Polyp-LDM、DiDGen、DiffAtlas 和 EBIS 进一步处理医学图像中的结构一致性、模态稀缺和边界质量。<a href="#ref-364">DiffuMask</a>、<a href="#ref-2166">DatasetDM</a>、<a href="#ref-2054">MatchMask</a>、<a href="#ref-2059">MRGen</a>、<a href="#ref-2039">Dual-Stream Diffusion</a>、<a href="#ref-2274">Polyp-LDM</a>、<a href="#ref-2035">DiDGen</a>、<a href="#ref-2276">DiffAtlas</a>、<a href="#ref-2272">EBIS</a>

医学分割的主要瓶颈不是生成图像是否逼真，而是病灶形状、边界和标注是否可信。D3S2、S²-RL 和 MRCA 还把分割模型反馈或数据集蒸馏纳入训练数据构造。<a href="#ref-2246">D3S2</a>、<a href="#ref-2283">S²-RL</a>、<a href="#ref-2120">MRCA</a>

### 3.4 表征学习

StableRep、SynCLR、Fake It Till You Make It 和 Scaling Laws 把合成图像当作视觉预训练数据，而不是只服务于一个下游分类器。合成规模、文本提示一致性、正样本构造方式和预训练目标共同决定迁移效果。合成数据可以在较少真实图像下扩大覆盖，但结果仍依赖生成器和语言监督。<a href="#ref-2258">StableRep</a>、<a href="#ref-2260">SynCLR</a>、<a href="#ref-2262">Fake It Till You Make It</a>、<a href="#ref-2072">Scaling Laws of Synthetic Images</a>

### 3.5 数据集蒸馏

数据集蒸馏把目标从“生成更多样本”改成“生成一组能代表训练信号的少量样本”。IMS3、EVLF、D3S2、S²-RL、DIVER 和 SFM 分别从分布聚合、视觉语言融合、分割蒸馏、强化学习和统计流匹配等角度压缩数据。蒸馏比例、初始化方式、IPC、验证集和下游骨干必须同时报告，蒸馏结果不能与普通增广的准确率直接横比。<a href="#ref-2106">IMS3</a>、<a href="#ref-2108">EVLF</a>、<a href="#ref-2246">D3S2</a>、<a href="#ref-2283">S²-RL</a>、<a href="#ref-2285">DIVER</a>、<a href="#ref-2287">SFM</a>

## 4. 按数据条件

条件集合是交叉索引，同一篇论文可以同时出现于多个条件分支。83 条证据中的交叉计数为：少样本与数据稀缺 22 条、长尾与类别不平衡 24 条、跨域与域偏移 16 条、标签稀缺与半监督 7 条、持续学习与鲁棒性 6 条。

### 4.1 少样本与数据稀缺

GIF、DataDream、DA-Fusion、Diff-II、BOB、ChimeraLoRA、LoFT、1S-DAug 和 Few-Shot Synthetic Data Generation 都围绕少量真实样本下的领域贴合、类内变化和生成预算展开。共同趋势是用轻量适配、图像条件或样本选择代替盲目扩大生成规模。<a href="#ref-2162">GIF</a>、<a href="#ref-2150">DataDream</a>、<a href="#ref-2158">DA-Fusion</a>、<a href="#ref-2118">Diff-II</a>、<a href="#ref-2102">BOB</a>、<a href="#ref-2100">ChimeraLoRA</a>、<a href="#ref-2281">LoFT</a>、<a href="#ref-2210">1S-DAug</a>、<a href="#ref-2202">Few-Shot Synthetic Data Generation</a>

### 4.2 长尾与类别不平衡

Fill-Up、DiffuLT、CORAL、DBG、SynBalance、DAR-GDA、Tail-SD、TRCGL-Net 和长尾医学分类工作表明，尾类数量平衡只是起点。更重要的是生成样本是否覆盖尾类的决策边界、姿态和背景变化。<a href="#ref-2170">Fill-Up</a>、<a href="#ref-351">DiffuLT</a>、<a href="#ref-2122">CORAL</a>、<a href="#ref-2098">DBG</a>、<a href="#ref-2254">SynBalance</a>、<a href="#ref-2130">DAR-GDA</a>、<a href="#ref-2186">Tail-SD</a>、<a href="#ref-2268">TRCGL-Net</a>、

### 4.3 跨域与域偏移

S3OD、DisCL、Free-Lunch、Pathology LDM-SSDA 和医学域泛化工作把合成数据用于跨数据集、跨设备或跨场景测试。跨域收益不能只看源域准确率，需要报告外部域、many/medium/few 或 OOD 指标。<a href="#ref-2104">S3OD</a>、<a href="#ref-2116">DisCL</a>、<a href="#ref-2244">Free-Lunch</a>、<a href="#ref-2224">Pathology LDM-SSDA</a>

### 4.4 标签稀缺与半监督

MatchMask、MRGen、Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning、EBIS 把生成过程用于缺少 mask、少量标注或未标注数据的场景。此类方法的关键证据是标注一致性和少标注曲线，而不是单独的 FID。<a href="#ref-2054">MatchMask</a>、<a href="#ref-2059">MRGen</a>、<a href="#ref-2256">Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning</a>、<a href="#ref-2272">EBIS</a>、

### 4.5 持续学习与鲁棒性

DiffAug、Class-Mixed Diffusion Augmentation、1S-DAug 和 Not Just Pretty Pictures 分别从 diffuse-and-denoise、快捷方式破除、少样本鲁棒性和干预式增广角度研究鲁棒性。持续学习和鲁棒性条件下，合成数据需要覆盖变化因素，同时避免把生成器自身的偏差带入训练集。<a href="#ref-2238">DiffAug</a>、<a href="#ref-2243">Class-Mixed Diffusion Augmentation</a>、<a href="#ref-2210">1S-DAug</a>、<a href="#ref-2252">Not Just Pretty Pictures</a>

## 5. 按方法阶段

方法阶段描述一条增广流水线中的动作，和下游任务、数据条件可以交叉组合。

### 5.1 图像条件编辑

DA-Fusion、DiffuseMix、Diff-Mix、Diff-II、DisCL 和 1S-DAug 以真实图像、反演潜变量或结构条件为起点，在去噪过程中控制外观、姿态和语义变化。它们通常更容易保持标签和布局，但变化范围受输入图像限制。<a href="#ref-2158">DA-Fusion</a>、<a href="#ref-123">DiffuseMix</a>、<a href="#ref-2154">Diff-Mix</a>、<a href="#ref-2118">Diff-II</a>、<a href="#ref-2116">DisCL</a>、<a href="#ref-2210">1S-DAug</a>

### 5.2 任务反馈生成

DBG、MRCA、C2I、S3OD 和用户引导的冠脉检测增广把分类器、分割器或检测器的边界、不确定性和梯度反馈给生成器。反馈闭环能把预算集中到模型缺口，但也会把反馈模型的错误放大。<a href="#ref-2098">DBG</a>、<a href="#ref-2120">MRCA</a>、<a href="#ref-2190">C2I</a>、<a href="#ref-2104">S3OD</a>、<a href="#ref-2279">Diffusion-Based User-Guided Data Augmentation</a>

### 5.3 图像—标注联合生成

DiffuMask、DatasetDM、MatchMask、MRGen、Dual-Stream Diffusion、Polyp-LDM、DiDGen 和 DiffAtlas 同时生成图像及 mask、depth 或其他稠密标注。这里的评价应同时包含图像质量、标注质量和下游模型收益。<a href="#ref-364">DiffuMask</a>、<a href="#ref-2166">DatasetDM</a>、<a href="#ref-2054">MatchMask</a>、<a href="#ref-2059">MRGen</a>、<a href="#ref-2039">Dual-Stream Diffusion</a>、<a href="#ref-2274">Polyp-LDM</a>、<a href="#ref-2035">DiDGen</a>、<a href="#ref-2276">DiffAtlas</a>

### 5.4 生成后筛选

Chamfer Guidance、Post-Generation Curation、BRIDGE 和 SynBalance 把样本质量、覆盖度、互补性或任务效用作为生成后的选择信号。结果显示，筛选常常比继续增加样本数量更稳定，但筛选器本身也可能引入偏差。<a href="#ref-2234">Chamfer Guidance</a>、<a href="#ref-2250">Post-Generation Curation</a>、<a href="#ref-2182">BRIDGE</a>、<a href="#ref-2254">SynBalance</a>


## 6. 统一评测与结果比较

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

| 方法或证据 | 新分类位置 | 论文报告的结果 | 主要解释 |
|:--|:--|:--|:--|
| <a href="#ref-116">TADA</a> | 按下游任务：分类 | 只增强训练早期未学到的 30%–40% 样本，最高提升 2.8 个百分点 | 选择样本可能比增加数量更重要 |
| <a href="#ref-2102">BOB</a> | 按下游任务：分类；按数据条件：少样本与数据稀缺 | 24 个设置中 18 个超过先前方法，14 个设置提升超过 2 个百分点 | 显式建模背景和姿态等类无关因素 |
| <a href="#ref-2054">MatchMask</a> | 按下游任务：分割与稠密预测 | PASCAL VOC 从 67.5% 提升到 74.3% mIoU | mask 中心生成、层自适应融合和相对过滤共同作用 |
| <a href="#ref-2166">DatasetDM</a> | 按下游任务：分割与稠密预测 | VOC 提升 13.3 mIoU，COCO 提升 12.1 AP | 感知解码器生成图像与标注配对数据 |
| <a href="#ref-2104">S3OD</a> | 按下游任务：目标检测 | synthetic-only 的跨数据集错误减少 20%–50% | 合成场景和歧义感知架构共同改善跨域泛化 |
| <a href="#ref-2120">MRCA</a> | 长尾实例分割 | 约 6% 的生成预算仍能获得持续收益 | 用实例分割器反馈生成不确定目标 |
| <a href="#ref-240">医学 DiffAug</a> | 医学分割 | 小息肉和扁平病灶的 false negative rate 最高降低 28%，推理约 12.5 samples/s | 文本引导、质量验证和快速潜变量估计共同影响结果 |
| <a href="#ref-2096">医学影像系统综述</a> | 多种医学任务 | synthetic-only 通常有限，real+synthetic 的分类收益较小，分割收益更小 | 合成数据更像真实数据的补充 |


## 6.1 数据集与结果的整体对照

论文使用的数据集可以分成几组。分类增广主要集中在 CIFAR、ImageNet、CUB、Aircraft、Flowers 和少样本自然图像。长尾工作常用 ImageNet-LT、CIFAR-LT、LVIS 和专门的医学长尾数据。分割与图像—掩码生成更多使用 PASCAL VOC、COCO、ADE20K，以及 ISIC、BUSI、PH2、DDTI 等医学数据。表中的结果保留论文原始指标，不能把不同任务的数值拼成一个总排行榜。

| 新分类位置 | 代表方法 | 主要数据集 | 论文报告的结果 | 比较时需要固定的因素 |
|:--|:--|:--|:--|:--|
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-116">TADA</a> | CIFAR-10/100、TinyImageNet、ImageNet | 只增广训练早期未学到的 30%–40% 样本，最高提升 2.8 个百分点 | 真实数据量、增广比例和骨干 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2102">BOB</a> | 多个细粒度分类数据集 | 24 个设置中 18 个超过先前方法，14 个设置提升超过 2 个百分点 | shot 数、属性建模和 T2I 模型 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2158">DA-Fusion</a> | Leafy Spurge、COCO、PASCAL VOC | 在少样本分类和真实杂草识别任务上提升准确率 | 数据集构造和每类样本数 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2150">DataDream</a> | 10 个少样本分类数据集，包括 ImageNet few-shot | 7/10 个数据集超过先前方法，其余接近或相当 | LoRA 预算和 shot 数 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2154">Diff-Mix</a> | CUB、FGVC-Aircraft、Flower | 在常规、少样本和长尾设置中稳定提升 | 类别粒度、生成倍数和下游训练策略 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2174">DIAGen</a> | 少标注分类与 OOD 设置 | 对 OOD 样本的收益比标准增广和 DA-Fusion 更明显 | OOD 定义和样本筛选权重 |
| 按下游任务：分类；按数据条件：少样本与数据稀缺 | <a href="#ref-2178">MGRAttn-RNO</a> | CUB-LT IF100、Flower-LT IF100 | 在 CUB-LT IF100 上最高提升 2.50 和 2.94 个百分点 | 不平衡比例和前景掩码质量 |
| 按下游任务：分类；按数据条件：长尾与类别不平衡 | <a href="#ref-2116">DisCL</a> | ImageNet-LT、iWildCam | ImageNet-LT 尾类准确率从 4.40% 提升到 23.64%，整体从 35.80% 提升到 39.82%；iWildCam OOD F1 从 35.5 提升到 38.2 | many/medium/few 划分和域外测试集 |
| 按下游任务：分类；按数据条件：长尾与类别不平衡 | <a href="#ref-2098">DBG</a>、<a href="#ref-2122">CORAL</a>、<a href="#ref-351">DiffuLT</a> | ImageNet-LT、CIFAR-LT 等 | 重点改善尾类准确率、尾类生成质量和决策边界分离 | imbalance factor、采样预算和 backbone |
| 按下游任务：目标检测/分割与稠密预测；按数据条件：长尾与类别不平衡 | <a href="#ref-2186">Tail-SD</a>、<a href="#ref-2120">MRCA</a> | COCO、LVIS、长尾实例分割基准 | Tail-SD 的 LVIS rare Box AP 提升 4.89，rare Mask AP 提升 5.68；MRCA 约用 6% 生成预算获得持续收益 | rare 类定义、实例预算和评测协议 |
| 按下游任务：分类；按数据条件：长尾与类别不平衡 | <a href="#ref-2268">TRCGL-Net</a> | PadChest | tail mAP 0.4904，overall mAP 0.4408，mAUC 0.8989 | 多标签指标不能与单标签分类直接比较 |
| 按下游任务：分割与稠密预测 | <a href="#ref-2054">MatchMask</a> | PASCAL VOC、COCO、ADE20K | PASCAL VOC mIoU 从 67.5% 提升到 74.3% | mask 对齐、标注稀缺比例和分割骨干 |
| 按方法阶段：图像—标注联合生成 | <a href="#ref-2166">DatasetDM</a> | PASCAL VOC 2012、COCO 2017 | VOC 提升 13.3 mIoU，COCO 提升 12.1 AP | 生成标注类型和下游模型 |
| 按下游任务：目标检测 | <a href="#ref-2104">S3OD</a> | DIS、HR-SOD | synthetic-only 跨数据集错误减少 20%–50% | 错误率与 mIoU、AP 不能混比 |
| 按方法阶段：图像—标注联合生成 | <a href="#ref-2056">MedDiff-FT</a> | ISIC-2017/2018、BUSI、PH2、DDTI | 正文报告五个医学分割数据集平均提升约 1% Dice，README 另有口径，需复核 | 模态、器官和标注比例 |
| 按下游任务：数据集蒸馏；按方法阶段：生成后筛选 | <a href="#ref-2246">D3S2</a>、<a href="#ref-2283">S²-RL</a> | ADE20K、COCO-Stuff、COCO | D3S2 在 1% 压缩率下 ADE20K/COCO-Stuff mIoU 为 24.99%/35.49%；S²-RL 使用 SegFormer-B5 评估 mIoU 和 mAcc | IPC、真实验证集和蒸馏预算 |
| 按下游任务：表征学习 | <a href="#ref-2258">StableRep</a>、<a href="#ref-2260">SynCLR</a> | 合成图像规模学习、ImageNet、ADE20K | StableRep 用 20M 合成图像超过 50M 真实图像训练的 CLIP；SynCLR 在 ADE20K 上比 MAE、iBOT 高 5.0、3.1 mIoU | 预训练规模、语言监督和下游协议 |
| 按方法阶段：生成后筛选 | <a href="#ref-2234">Chamfer Guidance</a> | 2 或 32 个真实 exemplar 的 few-shot 设置 | 2 exemplar 时 precision 96.4%、coverage 86.4%；32 exemplar 时 precision 97.5%、coverage 92.7%；下游 ID/OOD accuracy 最高提升 15/16% | 质量指标与下游效用需同时报告 |

这些结果说明，合成数据的收益通常来自数据条件和任务模型的匹配。少样本分类更关注类内变化和分布贴合，长尾方法更关注尾类覆盖与决策边界，分割方法还必须验证图像和掩码的一致性。跨论文比较时应优先复现统一评测框架，再比较同一任务、同一数据预算和同一指标下的结果。

## 7. 局限和风险

**语义和分布偏移。** 大型自然图像模型并不自动理解医学形态、传感器噪声或三维解剖结构。提示错误会生成漂亮但标签不对的图像。

**保真度和多样性的冲突。** 强图像条件提高标签稳定性，却可能只复制训练集。弱条件增加变化，却会带来背景捷径和类别混淆。

**反馈闭环偏差。** 分类器、分割器或奖励模型既负责挑样本又影响生成器，可能把自身错误反复放大。

**计算和可复现性。** 反演、微调、多步采样、过滤和多轮反馈的成本很高。许多论文没有完整报告 prompt、随机种子、合成比例、过滤阈值和生成预算。

**隐私和记忆。** 预训练扩散模型可能复现训练数据中的人物、版权内容或敏感模式。合成图像也可能把数据集偏见放大到下游模型。

**外部验证不足。** 多数工作只在少量公开数据集和单一下游骨干上验证，跨医院、跨设备、跨地域的证据仍然有限。

## 8. 开源状况

当前按本综述 83 条证据记录统计，其中 47 条正文出现 GitHub URL。这个数字包含依赖库和基础模型链接，因此需要进一步区分作者仓库、依赖项目和基础模型。下面只把论文来源页或条目中明确指向的作者仓库列为“代码路径”。链接指向仓库根目录，进入仓库后的 `README`、`configs`、`scripts` 和训练入口通常能看到复现实验所需的具体命令。33 个作者仓库链接中，31 个可以访问，29 个已经完成本地浅克隆，IGD-ML 和 LoFT 的克隆仍需补齐，TRCGL-Net 和 SFD-Polyp 的链接返回 Repository not found。代码归档目录为 `C:\Users\Richard\Documents\Codex\2026-11-05\s\outputs\diffusion_augmentation_code`，依赖库和基础模型没有混入这个目录。

| 方法 | 作者代码路径 | 备注 |
|:--|:--|:--|
| DiffDA-Eval | [nukezil/DiffDA-Eval](https://github.com/nukezil/DiffDA-Eval) | 统一评测与复现实验 |
| DBG | [keepdigitalabc-svg/DBG](https://github.com/keepdigitalabc-svg/DBG) | 决策边界生成 |
| IMS3 | [Westlake-AGI-Lab/IMS3](https://github.com/Westlake-AGI-Lab/IMS3) | 扩散数据集蒸馏 |
| Learnability Gap | [MischaD/LearnabilityGap](https://github.com/MischaD/LearnabilityGap) | 医学潜空间扩散分析 |
| DisCL | [tianyi-lab/DisCL](https://github.com/tianyi-lab/DisCL) | 合成到真实的课程学习 |
| Diff-II | [HKUST-LongGroup/Diff-II](https://github.com/HKUST-LongGroup/Diff-II) | 反演圆插值增广 |
| CORAL | [SankarLab/coral-lt-diffusion](https://github.com/SankarLab/coral-lt-diffusion) | 长尾潜表示解耦 |
| IGD-ML | [pranoydas51/IGD-ML](https://github.com/pranoydas51/IGD-ML) | 长尾生成建模 |
| DAR-GDA | [ruyianry/gda-dar](https://github.com/ruyianry/gda-dar) | 蒸馏、对齐与重加权 |
| Diffusing DeBias | [Malga-Vision/DiffusingDeBias](https://github.com/Malga-Vision/DiffusingDeBias) | 偏置放大与去偏 |
| MedDiff-FT | [JianhaoXie1/MedDiff-FT](https://github.com/JianhaoXie1/MedDiff-FT) | 医学图像轻量微调 |
| DataDream | [ExplainableML/DataDream](https://github.com/ExplainableML/DataDream) | 少样本数据集生成 |
| Diff-Mix | [Zhicaiwww/Diff-Mix](https://github.com/Zhicaiwww/Diff-Mix) | 跨类别扩散混合 |
| DA-Fusion | [brandontrabucco/da-fusion](https://github.com/brandontrabucco/da-fusion) | 图像条件扩散增广 |
| GIF | [Vanint/DatasetExpansion](https://github.com/Vanint/DatasetExpansion) | 小规模数据集扩展 |
| DIAGen | [visinf/DIAGen](https://github.com/visinf/DIAGen) | 生成式图像增广 |
| MGRAttn-RNO | [hvcl/MGRAttn-RNO](https://github.com/hvcl/MGRAttn-RNO) | 参考注意力与排斥噪声 |
| 病理图像 LDM-SSDA | [hsu-lab/ldm-pathology-ssda](https://github.com/hsu-lab/ldm-pathology-ssda) | 半监督域适配 |
| SynRep / SynCLR | [google-research/syn-rep-learn](https://github.com/google-research/syn-rep-learn) | 合成表示学习 |
| DiffAug | [oore-lab/diffaug](https://github.com/oore-lab/diffaug) | diffuse-and-denoise 增广 |
| Free-Lunch | [zzzzj311-droid/Free-Lunch-SITN](https://github.com/zzzzj311-droid/Free-Lunch-SITN) | 跨域少样本检测 |
| D3S2 | [zwj084/D3S2](https://github.com/zwj084/D3S2) | 语义分割数据集蒸馏 |
| LDMLR | [AlvinHan123/LDMLR](https://github.com/AlvinHan123/LDMLR) | 长尾特征扩散 |
| TRCGL-Net | [November-1113/TRCG](https://github.com/November-1113/TRCG) | 胸片长尾分类 |
| SFD-Polyp | [zhaiying0925/SFD-](https://github.com/zhaiying0925/SFD-) | 息肉图像—掩码生成 |
| EBIS | [VEDAs-Lab/EBIS](https://github.com/VEDAs-Lab/EBIS) | 烧伤分割与扩散教师 |
| Polyp-LDM | [16rq/Polyp-LDM](https://github.com/16rq/Polyp-LDM) | 息肉边界对齐生成 |
| DiDGen | [junjie-shentu/DiDGen](https://github.com/junjie-shentu/DiDGen) | 皮肤镜双任务生成 |
| DiffAtlas | [M3DV/DiffAtlas](https://github.com/M3DV/DiffAtlas) | 图谱分割图像—掩码生成 |
| DiGDA | [medipixel/DiGDA](https://github.com/medipixel/DiGDA) | 冠脉检测用户引导增广 |
| LoFT | [ExplainableML/LoFT](https://github.com/ExplainableML/LoFT) | LoRA 少样本数据生成 |
| S2-RL | [00why00/S2RL](https://github.com/00why00/S2RL) | 分割数据集蒸馏 |
| SFM | [einsteinxia/SFM](https://github.com/einsteinxia/SFM) | 自监督数据集蒸馏 |

部分论文只提供了基础模型或依赖库链接，例如 `diffusers`、`latent-diffusion`、`mmsegmentation`、`detectron2` 和 `FLUX`，不能据此判断论文作者实现已经公开。当前尚未定位到作者仓库的代表性方法包括 S3OD、DatasetDM、Fill-Up、Tail-SD、RealSynz、SaSPA、Chamfer Guidance、MRGen、C2I 和 DIVER。仓库是否包含预训练权重、生成数据和完整训练脚本，还需要以仓库当前内容为准。

## 9. 仍需补全文本的条目

以下两条目前仍保留在 Zotero 中，但没有本地 PDF。文章中的方法和结果依据题目、摘要或出版页面，属于待补全文献。

- Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data
- Class-Mixed Diffusion Augmentation for Shortcut-Breaking in Continual Learning

## 10. 一句话总结

扩散模型数据增广正在从“生成更多图像”转向“生成对特定模型和特定缺口有用的图像”。下一步的关键在于建立可复现的任务效用评测、可靠的样本筛选、结构一致的标注生成和跨域外部验证。



## 阅读边界

本页使用的主体证据来自 Zotero 本地 PDF 和条目中的论文来源链接。2 篇没有本地 PDF 的记录已经明确标注为“当前无本地 PDF”，它们在当前版本中作为待补全文献处理。正式接收、预印本、workshop 和 OpenReview 在审状态也按 Zotero 条目和来源页面分别保留。

## 参考文献

<details class="references-fold">
<summary>参考文献（点击展开）</summary>

<ul class="refs">
<li id="ref-bg-gan2014"><span class="ref-no">[1]</span> Goodfellow et al. (2014). <strong>Generative Adversarial Nets.</strong> <a href="https://arxiv.org/abs/1406.2661">来源</a></li>
<li id="ref-bg-ddpm2020"><span class="ref-no">[2]</span> Ho et al. (2020). <strong>Denoising Diffusion Probabilistic Models.</strong> <a href="https://arxiv.org/abs/2006.11239">来源</a></li>
<li id="ref-bg-ldm2022"><span class="ref-no">[3]</span> Rombach et al. (2022). <strong>High-Resolution Image Synthesis with Latent Diffusion Models.</strong> <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html">来源</a></li>
<li id="ref-bg-controlnet2023"><span class="ref-no">[4]</span> Zhang et al. (2023). <strong>Adding Conditional Control to Text-to-Image Diffusion Models.</strong> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html">来源</a></li>
<li id="ref-bg-unidiffda2026"><span class="ref-no">[5]</span> Li et al. (2026). <strong>Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation.</strong> <a href="https://arxiv.org/abs/2603.08364">来源</a></li>
<li id="ref-bg-tadasurvey2026"><span class="ref-no">[6]</span> When More Data Hurts (2026). <strong>When More Data Hurts: A Critical Survey of Failure Modes in Generative Data Augmentation for Image Classification.</strong> <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6448800">来源</a></li>
<li id="ref-2090"><span class="ref-no">[7]</span> <strong>Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation</strong>（2026；IJCV 2026；分类、综述/评测；全文PDF已读取；<a href="https://arxiv.org/abs/2603.08364">来源</a>；<a href="https://github.com/huggingface/accelerate">代码/项目</a>；<a href="https://github.com/nukezil/DiffDA-Eval">代码/项目</a>）</li>
<li id="ref-2092"><span class="ref-no">[8]</span> <strong>Advances in Diffusion Models for Image Data Augmentation: A Review of Methods, Models, Evaluation Metrics and Future Research Directions</strong>（2025；Artificial Intelligence Review 2025；综述/评测；全文PDF已读取；<a href="https://doi.org/10.1007/s10462-025-11116-x">来源</a>）</li>
<li id="ref-2094"><span class="ref-no">[9]</span> <strong>A Comprehensive Survey on Data Augmentation</strong>（2026；IEEE TKDE 2026；综述/评测；全文PDF已读取；<a href="https://people.cs.vt.edu/~clu/Publication/2026/Survey_on_Data_Augmentation-2026.pdf">来源</a>）</li>
<li id="ref-2096"><span class="ref-no">[10]</span> <strong>Quality Assessment and Model Training Impact of Synthetic Images in Medical Diagnostic Imaging: Systematic Review</strong>（2026；JMIR AI 2026；医学影像、综述/评测；全文PDF已读取；<a href="https://ai.jmir.org/2026/1/e81472/">来源</a>；<a href="https://github.com/DS3-SisCat/2507-synthetic-images-sysrev/">代码/项目</a>；<a href="https://github.com/cebudding/MedNeurips_">代码/项目</a>）</li>
<li id="ref-2097"><span class="ref-no">[11]</span> <strong>When More Data Hurts: A Critical Survey of Failure Modes in Generative Data Augmentation for Image Classification</strong>（2026；分类、综述/评测；全文PDF已读取；<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6448800">来源</a>）</li>
<li id="ref-116"><span class="ref-no">[12]</span> <strong>Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models</strong>（2026；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html">来源</a>；<a href="https://github.com/lucidrains/vit-pytorch.">代码/项目</a>；<a href="https://github.com/openai/glide-text2im.">代码/项目</a>）</li>
<li id="ref-2098"><span class="ref-no">[13]</span> <strong>Decision Boundary-aware Generation for Long-tailed Learning (DBG)</strong>（2026；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Decision_Boundary-aware_Generation_for_Long-tailed_Learning_CVPR_2026_paper.html">来源</a>；<a href="https://github.com/keepdigitalabc-svg/DBG.">代码/项目</a>）</li>
<li id="ref-2100"><span class="ref-no">[14]</span> <strong>ChimeraLoRA: Multi-Head LoRA-Guided Synthetic Datasets</strong>（2026；分类、少样本与数据稀缺、长尾与类别不平衡；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ChimeraLoRA_Multi-Head_LoRA-Guided_Synthetic_Datasets_CVPR_2026_paper.html">来源</a>）</li>
<li id="ref-2102"><span class="ref-no">[15]</span> <strong>Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification (BOB)</strong>（2026；分类、细粒度、少样本与数据稀缺；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Beyond_Objects_Contextual_Synthetic_Data_Generation_for_Fine-Grained_Classification_CVPR_2026_paper.html">来源</a>）</li>
<li id="ref-2054"><span class="ref-no">[16]</span> <strong>MatchMask: Mask-Centric Generative Data Augmentation for Label-Scarce Semantic Segmentation</strong>（2026；分割与稠密预测、少标注；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Lin_MatchMask_Mask-Centric_Generative_Data_Augmentation_for_Label-Scarce_Semantic_Segmentation_CVPR_2026_paper.html">来源</a>）</li>
<li id="ref-2104"><span class="ref-no">[17]</span> <strong>S3OD: Towards Generalizable Salient Object Detection with Synthetic Data</strong>（2026；显著性检测、跨域与域偏移；全文PDF已读取；<a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/ead7401f3a05d5285f49a915cfac2f2a-Abstract-Conference.html">来源</a>；<a href="https://github.com/black-forest-labs/flux">代码/项目</a>；<a href="https://github.com/black-forest-labs/flux.">代码/项目</a>）</li>
<li id="ref-2106"><span class="ref-no">[18]</span> <strong>IMS3: Breaking Distributional Aggregation in Diffusion-Based Dataset Distillation</strong>（2026；数据集蒸馏、分类；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Wang_IMS3_Breaking_Distributional_Aggregation_in_Diffusion-Based_Dataset_Distillation_CVPR_2026_paper.html">来源</a>；<a href="https://github.com/Westlake-AGI-Lab/IMS3.">代码/项目</a>）</li>
<li id="ref-2108"><span class="ref-no">[19]</span> <strong>EVLF: Early Vision-Language Fusion for Generative Dataset Distillation</strong>（2026；数据集蒸馏、分类；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Cai_EVLF_Early_Vision-Language_Fusion_for_Generative_Dataset_Distillation_CVPR_2026_paper.html">来源</a>；<a href="https://github.com/Guang000/">代码/项目</a>）</li>
<li id="ref-2110"><span class="ref-no">[20]</span> <strong>WILD-Diffusion: A WDRO Inspired Training Method for Diffusion Models under Limited Data</strong>（2026；生成模型训练、少样本与数据稀缺；全文PDF已读取；<a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/9f709622912e87f8d84bffb097ae5503-Abstract-Conference.html">来源</a>）</li>
<li id="ref-2112"><span class="ref-no">[21]</span> <strong>The Learnability Gap in Medical Latent Diffusion</strong>（2026；医学影像、长尾与类别不平衡、分类；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2026/1053-Paper3049.html">来源</a>；<a href="https://github.com/MischaD/LearnabilityGap.git.">代码/项目</a>）</li>
<li id="ref-2114"><span class="ref-no">[22]</span> <strong>Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation</strong>（2026；医学影像、3D生成、分割与稠密预测；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2026/html/An_Sketch2CT_Multimodal_Diffusion_for_Structure-Aware_3D_Medical_Volume_Generation_CVPR_2026_paper.html">来源</a>）</li>
<li id="ref-2116"><span class="ref-no">[23]</span> <strong>Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion (DisCL)</strong>（2025；分类、长尾与类别不平衡、低质量数据；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.html">来源</a>；<a href="https://github.com/tianyi-lab/DisCL">代码/项目</a>）</li>
<li id="ref-2118"><span class="ref-no">[24]</span> <strong>Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification (Diff-II)</strong>（2025；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Inversion_Circle_Interpolation_Diffusion-based_Image_Augmentation_for_Data-scarce_Classification_CVPR_2025_paper.html">来源</a>；<a href="https://github.com/HKUST-LongGroup/Diff-II.">代码/项目</a>）</li>
<li id="ref-2120"><span class="ref-no">[25]</span> <strong>Sample-Efficient Multi-Round Generative Data Augmentation for Long-Tail Instance Segmentation (MRCA)</strong>（2025；实例分割与稠密预测、长尾与类别不平衡；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/69f59f96c3aae0ba9f463e7903715e1e-Abstract-Conference.html">来源</a>）</li>
<li id="ref-2122"><span class="ref-no">[26]</span> <strong>CORAL: Disentangling Latent Representations in Long-Tailed Diffusion</strong>（2025；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/a32539cb16274581a17e679f6046f4bf-Abstract-Conference.html">来源</a>；<a href="https://github.com/SankarLab/coral-lt-diffusion.">代码/项目</a>）</li>
<li id="ref-2126"><span class="ref-no">[27]</span> <strong>Principled Long-Tailed Generative Modeling via Diffusion Models</strong>（2025；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/9ff5493d7d4a4b4002d67738237abd5b-Abstract-Conference.html">来源</a>；<a href="https://github.com/pranoydas51/IGD-ML">代码/项目</a>）</li>
<li id="ref-2130"><span class="ref-no">[28]</span> <strong>Generative Data Augmentation via Diffusion Distillation, Adversarial Alignment, and Importance Reweighting (DAR-GDA)</strong>（2025；分类、效率/分布对齐；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/62e5721247075dd097023d077d8e22f7-Abstract-Conference.html">来源</a>；<a href="https://github.com/ruyianry/gda-dar.">代码/项目</a>）</li>
<li id="ref-2134"><span class="ref-no">[29]</span> <strong>Diffusing DeBias: Synthetic Bias Amplification for Model Debiasing</strong>（2025；分类、去偏；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/a982938f457eafffedc3205c46b4bac7-Abstract-Conference.html">来源</a>；<a href="https://github.com/Malga-Vision/DiffusingDeBias.">代码/项目</a>）</li>
<li id="ref-2138"><span class="ref-no">[30]</span> <strong>AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition</strong>（2025；分类、人脸识别；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html">来源</a>）</li>
<li id="ref-2142"><span class="ref-no">[31]</span> <strong>Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models</strong>（2025；稠密预测、分割与稠密预测、多任务；全文PDF已读取；<a href="https://proceedings.iclr.cc/paper_files/paper/2025/hash/1ec299a5229034141e58aeded0d0b9de-Abstract-Conference.html">来源</a>）</li>
<li id="ref-2056"><span class="ref-no">[32]</span> <strong>MedDiff-FT: Data-Efficient Diffusion Model Fine-Tuning with Structural Guidance for Controllable Medical Image Synthesis</strong>（2025；医学影像、分割与稠密预测、图像—标注联合生成；全文PDF已读取；<a href="https://doi.org/10.1007/978-3-032-04965-0_29">来源</a>；<a href="https://github.com/JianhaoXie1/MedDiff-FT.">代码/项目</a>）</li>
<li id="ref-2146"><span class="ref-no">[33]</span> <strong>Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification</strong>（2025；分类、长尾与类别不平衡、食品；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2025W/MTF/html/Koh_Synthetic_Data_Augmentation_using_Pre-trained_Diffusion_Models_for_Long-tailed_Food_CVPRW_2025_paper.html">来源</a>）</li>
<li id="ref-2150"><span class="ref-no">[34]</span> <strong>DataDream: Few-shot Guided Dataset Generation</strong>（2024；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08991.pdf">来源</a>；<a href="https://github.com/ExplainableML/DataDream.">代码/项目</a>）</li>
<li id="ref-2154"><span class="ref-no">[35]</span> <strong>Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model (Diff-Mix)</strong>（2024；分类、细粒度；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Enhance_Image_Classification_via_Inter-Class_Image_Mixup_with_Diffusion_Model_CVPR_2024_paper.html">来源</a>；<a href="https://github.com/Zhicaiwww/Diff-Mix">代码/项目</a>；<a href="https://github.com/huggingface/">代码/项目</a>）</li>
<li id="ref-123"><span class="ref-no">[36]</span> <strong>DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models</strong>（2024；分类、持续学习与鲁棒性；全文PDF已读取；<a href="https://doi.org/10.1109/cvpr52733.2024.02608">来源</a>）</li>
<li id="ref-2158"><span class="ref-no">[37]</span> <strong>Effective Data Augmentation With Diffusion Models (DA-Fusion)</strong>（2024；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://proceedings.iclr.cc/paper_files/paper/2024/file/3f1351a10a3f904e6f3845a94275a909-Paper-Conference.pdf">来源</a>；<a href="https://github.com/brandontrabucco/da-fusion.">代码/项目</a>；<a href="https://github.com/rom1504/clip-retrieval">代码/项目</a>）</li>
<li id="ref-2162"><span class="ref-no">[38]</span> <strong>Expanding Small-Scale Datasets with Guided Imagination (GIF)</strong>（2023；数据集扩增、分类、医学影像；全文PDF已读取；<a href="https://papers.neurips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html">来源</a>；<a href="https://github.com/Vanint/DatasetExpansion.">代码/项目</a>）</li>
<li id="ref-364"><span class="ref-no">[39]</span> <strong>DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models</strong>（2023；分割与稠密预测、开放词汇；全文PDF已读取；<a href="https://ieeexplore.ieee.org/document/10376904/">来源</a>）</li>
<li id="ref-2166"><span class="ref-no">[40]</span> <strong>DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models</strong>（2023；分割与稠密预测、实例分割与稠密预测、深度估计、数据集构建；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html">来源</a>；<a href="https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox">代码/项目</a>）</li>
<li id="ref-2170"><span class="ref-no">[41]</span> <strong>Fill-Up: Balancing Long-Tailed Data with Generative Models</strong>（2023；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://arxiv.org/abs/2306.07200">来源</a>；<a href="https://github.com/LambdaLabsML/lambda-diffusers">代码/项目</a>；<a href="https://github.com/Stability-AI/stablediffusion">代码/项目</a>；<a href="https://github.com/rinongal/textual_inversion/">代码/项目</a>；<a href="https://github.com/runwayml/stable-diffusion">代码/项目</a>）</li>
<li id="ref-2174"><span class="ref-no">[42]</span> <strong>DIAGen: Diverse Image Augmentation with Generative Models</strong>（2024；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://arxiv.org/abs/2408.14584">来源</a>；<a href="https://github.com/visinf/DIAGen">代码/项目</a>）</li>
<li id="ref-2178"><span class="ref-no">[43]</span> <strong>Improving Diffusion-Based Augmentation for Data-Scarce Image Classification via Mask-Guided Reference Attention and Repulsive Noise Optimization</strong>（2026；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://arxiv.org/abs/2611.04242">来源</a>；<a href="https://github.com/hvcl/MGRAttn-RNO">代码/项目</a>）</li>
<li id="ref-2182"><span class="ref-no">[44]</span> <strong>Balancing Representations by Identifying and Generating Under-represented Data (BRIDGE)</strong>（2026；分类、长尾与类别不平衡、无标签反馈；全文PDF已读取；<a href="https://arxiv.org/abs/2611.02952">来源</a>）</li>
<li id="ref-2186"><span class="ref-no">[45]</span> <strong>Improving Long-Tail Detection via Layout-Instruction Conditioned Diffusion Synthesis (Tail-SD)</strong>（2026；目标检测、长尾与类别不平衡；全文PDF已读取；<a href="https://arxiv.org/abs/2611.07553">来源</a>；<a href="https://github.com/ChenyunWu/PhraseCutDataset">代码/项目</a>；<a href="https://github.com/facebookresearch/detectron2">代码/项目</a>；<a href="https://github.com/yoctta/XPaste/issues/2">代码/项目</a>）</li>
<li id="ref-2190"><span class="ref-no">[46]</span> <strong>Generate What Matters: Steering Diffusion Models for Targeted Data Generation to Improve Classification (C2I)</strong>（2026；分类、任务反馈生成；全文PDF已读取；<a href="https://arxiv.org/abs/2607.12464">来源</a>）</li>
<li id="ref-2194"><span class="ref-no">[47]</span> <strong>Limitations of Synthetic Data Generation in Specialized Data-Scarce Domains</strong>（2026；分类、失败分析；全文PDF已读取；<a href="https://arxiv.org/abs/2608.13729">来源</a>）</li>
<li id="ref-2198"><span class="ref-no">[48]</span> <strong>Synthetic Data Generation for Long-Tail Medical Image Classification: A Case Study in Skin Lesions</strong>（2026；医学影像、分类、长尾与类别不平衡；全文PDF已读取；<a href="https://arxiv.org/abs/2605.03221">来源</a>）</li>
<li id="ref-2202"><span class="ref-no">[49]</span> <strong>Few-Shot Synthetic Data Generation with Diffusion Models for Downstream Vision Tasks</strong>（2026；分类、少样本与数据稀缺；全文PDF已读取；<a href="https://arxiv.org/abs/2605.11898">来源</a>）</li>
<li id="ref-2206"><span class="ref-no">[50]</span> <strong>Usefulness of Quantile-Aware Diffusion Modeling for Highly Imbalanced Tabular Data</strong>（2026；表格数据、长尾与类别不平衡；全文PDF已读取；<a href="https://arxiv.org/abs/2610.05825">来源</a>）</li>
<li id="ref-2210"><span class="ref-no">[51]</span> <strong>1S-DAug: One-Shot Data Augmentation for Robust Few-Shot Generalization</strong>（2026；分类、少样本与数据稀缺、测试时增广；全文PDF已读取；<a href="https://arxiv.org/abs/2602.00114">来源</a>）</li>
<li id="ref-2214"><span class="ref-no">[52]</span> <strong>Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation</strong>（2026；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://arxiv.org/abs/2602.04749">来源</a>）</li>
<li id="ref-2218"><span class="ref-no">[53]</span> <strong>RealSynz: Naturalizing Synthetic Images with Envelope Variation Statistics</strong>（2026；通用后处理、真实性/自然化；全文PDF已读取；<a href="https://openreview.net/challenge?redirect=%2Fforum%3Fid%3D80jkP3BMjb">来源</a>；<a href="https://github.com/black-forest-labs/flux">代码/项目</a>）</li>
<li id="ref-2224"><span class="ref-no">[54]</span> <strong>Semi-Supervised Domain Adaptation with Latent Diffusion for Pathology Image Classification</strong>（2026；IEEE JBHI 2026；医学影像、病理、域适应、分类；全文PDF已读取；<a href="https://doi.org/10.1109/JBHI.2026.3727978">来源</a>；<a href="https://github.com/hsu-lab/ldm-pathology-ssda.">代码/项目</a>）</li>
<li id="ref-2227"><span class="ref-no">[55]</span> <strong>Controlling Augmentations for Balancing Invariance and Diversification in One-Shot Image Classification</strong>（2026；Neurocomputing 2026；分类、少样本与数据稀缺、一-shot；全文PDF已读取；<a href="https://doi.org/10.1016/j.neucom.2026.134322">来源</a>；<a href="https://github.com/kkyuhun94/">代码/项目</a>）</li>
<li id="ref-351"><span class="ref-no">[56]</span> <strong>DiffuLT: Diffusion for Long-tail Recognition Without External Knowledge</strong>（2024；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://doi.org/10.52202/079017-3909">来源</a>）</li>
<li id="ref-2072"><span class="ref-no">[57]</span> <strong>Scaling Laws of Synthetic Images for Model Training ... for Now</strong>（2024；表征学习、合成数据规模律；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Scaling_Laws_of_Synthetic_Images_for_Model_Training_..._for_CVPR_2024_paper.html">来源</a>；<a href="https://github.com/google-research/syn-rep-learn">代码/项目</a>；<a href="https://github.com/openai/CLIP/tree/main/notebooks">代码/项目</a>）</li>
<li id="ref-2234"><span class="ref-no">[58]</span> <strong>Increasing the Utility of Synthetic Images through Chamfer Guidance</strong>（2025；表征学习、合成数据筛选；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/cd08b0700a0a4f4856c70d9346b3a506-Abstract-Conference.html">来源</a>；<a href="https://github.com/facebookresearch/dinov2">代码/项目</a>；<a href="https://github.com/huggingface/diffusers">代码/项目</a>；<a href="https://github.com/layer6ai-labs/dgm-eval">代码/项目</a>；<a href="https://github.com/openai/CLIP">代码/项目</a>）</li>
<li id="ref-2236"><span class="ref-no">[59]</span> <strong>SaSPA: Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation</strong>（2024；分类、细粒度；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/27c8b849acba6793f0b73f7ee7ea7397-Abstract-Conference.html">来源</a>；<a href="https://github.com/huggingface/diffusers">代码/项目</a>）</li>
<li id="ref-2238"><span class="ref-no">[60]</span> <strong>DiffAug: A Diffuse-and-Denoise Augmentation for Training Robust Classifiers</strong>（2024；分类、持续学习与鲁棒性；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/24e8b46430df965674221665816a4964-Paper-Conference.pdf">来源</a>；<a href="https://github.com/oore-lab/diffaug">代码/项目</a>）</li>
<li id="ref-2240"><span class="ref-no">[61]</span> <strong>Synthetic Data from Diffusion Models Improves ImageNet Classification</strong>（2023；Transactions on Machine Learning Research；分类、表征学习；全文PDF已读取；<a href="https://arxiv.org/abs/2304.08466">来源</a>）</li>
<li id="ref-2242"><span class="ref-no">[62]</span> <strong>Generative Active Learning via Bayesian Acquisition for Improving the Efficiency of Synthetic Data</strong>（2026；分类、主动学习；当前无本地PDF；<a href="https://neurips.cc/virtual/2026/loc/sydney/poster/149816">来源</a>）</li>
<li id="ref-2243"><span class="ref-no">[63]</span> <strong>Class-Mixed Diffusion Augmentation for Shortcut-Breaking in Continual Learning</strong>（2026；持续学习、分类；当前无本地PDF；<a href="https://neurips.cc/virtual/2026/loc/atlanta/poster/154098">来源</a>）</li>
<li id="ref-2244"><span class="ref-no">[64]</span> <strong>Free-Lunch Augmentation by Revisiting Diffusion-Based Data Generation for Cross-Domain Few-Shot Object Detection</strong>（2026；目标检测、跨域少样本与数据稀缺；全文PDF已读取；<a href="https://arxiv.org/abs/2608.04394">来源</a>；<a href="https://github.com/zzzzj311-droid/Free-Lunch-SITN.">代码/项目</a>）</li>
<li id="ref-2246"><span class="ref-no">[65]</span> <strong>D3S2: Diffusion-Guided Dataset Distillation for Semantic Segmentation</strong>（2026；分割与稠密预测、数据集蒸馏；全文PDF已读取；<a href="https://arxiv.org/abs/2605.25022">来源</a>；<a href="https://github.com/open-mmlab/mmsegmentation">代码/项目</a>；<a href="https://github.com/zwj084/D3S2.">代码/项目</a>）</li>
<li id="ref-2248"><span class="ref-no">[66]</span> <strong>FrozenDrive: Zero-Shot Text-Guided Driving Scene Generation and Data Augmentation with Parameter-Free Frozen Diffusion Model</strong>（2026；自动驾驶、目标检测、场景生成；全文PDF已读取；<a href="https://arxiv.org/abs/2606.20110">来源</a>）</li>
<li id="ref-2250"><span class="ref-no">[67]</span> <strong>Post-Generation Curation of Synthetic Images via Homogeneous-Heterogeneous Splitting</strong>（2026；合成数据筛选、分类；全文PDF已读取；<a href="https://arxiv.org/abs/2607.02637">来源</a>）</li>
<li id="ref-2252"><span class="ref-no">[68]</span> <strong>Not Just Pretty Pictures: Toward Interventional Data Augmentation Using Text-to-Image Generators</strong>（2024；分类、因果/干预增广；全文PDF已读取；<a href="https://proceedings.mlr.press/v235/yuan24e.html">来源</a>）</li>
<li id="ref-2254"><span class="ref-no">[69]</span> <strong>SynBalance: Harnessing Synthetic Data in Long-tailed Recognition</strong>（2025；分类、长尾与类别不平衡；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Jiang_SynBalance_Harnessing_Synthetic_Data_in_Long-tailed_Recognition_ICCVW_2025_paper.html">来源</a>）</li>
<li id="ref-276"><span class="ref-no">[70]</span> <strong>Latent-based Diffusion Model for Long-tailed Recognition</strong>（2024；分类、长尾与类别不平衡、特征增广；全文PDF已读取；<a href="https://doi.org/10.1109/cvprw63382.2024.00270">来源</a>；<a href="https://github.com/AlvinHan123/LDMLR">代码/项目</a>）</li>
<li id="ref-2256"><span class="ref-no">[71]</span> <strong>Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning</strong>（2024；标签稀缺与半监督学习、开放集、分类；全文PDF已读取；<a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/777cc2af0ab984e6dc48d168ce7b754f-Paper-Conference.pdf">来源</a>；<a href="https://github.com/CompVis/latent-diffusion">代码/项目</a>；<a href="https://github.com/LuChengTHU/dpm-solver">代码/项目</a>）</li>
<li id="ref-2258"><span class="ref-no">[72]</span> <strong>StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners</strong>（2023；表征学习、合成数据；全文PDF已读取；<a href="https://papers.neurips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html">来源</a>；<a href="https://github.com/google-research/syn-rep-learn">代码/项目</a>）</li>
<li id="ref-2260"><span class="ref-no">[73]</span> <strong>Learning Vision from Models Rivals Learning Vision from Data (SynCLR)</strong>（2024；表征学习、合成数据；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Learning_Vision_from_Models_Rivals_Learning_Vision_from_Data_CVPR_2024_paper.html">来源</a>；<a href="https://github.com/google-research/syn-rep-learn">代码/项目</a>）</li>
<li id="ref-2262"><span class="ref-no">[74]</span> <strong>Fake It Till You Make It: Learning Transferable Representations From Synthetic ImageNet Clones</strong>（2023；表征学习、合成数据；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2023/html/Sariyildiz_Fake_It_Till_You_Make_It_Learning_Transferable_Representations_From_CVPR_2023_paper.html">来源</a>）</li>
<li id="ref-2264"><span class="ref-no">[75]</span> <strong>Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning</strong>（2025；视觉-语言、分类；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Beyond_Words_Augmenting_Discriminative_Richness_via_Diffusions_in_Unsupervised_Prompt_CVPR_2025_paper.html">来源</a>）</li>
<li id="ref-2268"><span class="ref-no">[76]</span> <strong>TRCGL-Net: A Long-Tailed Multi-Label Chest X-Ray Classification Framework with Generative Data Augmentation and Label Co-Occurrence Modeling</strong>（2026；医学影像、胸片、长尾与类别不平衡、分类；全文PDF已读取；<a href="https://arxiv.org/abs/2607.00975">来源</a>；<a href="https://github.com/November-1113/TRCG">代码/项目</a>）</li>
<li id="ref-2059"><span class="ref-no">[77]</span> <strong>MRGen: Segmentation Data Engine For Underrepresented MRI Modalities</strong>（2025；医学影像、MRI、分割与稠密预测；全文PDF已读取；<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MRGen_Segmentation_Data_Engine_For_Underrepresented_MRI_Modalities_ICCV_2025_paper.html">来源</a>）</li>
<li id="ref-2270"><span class="ref-no">[78]</span> <strong>Learning Diverse and Realistic Polyp Data via Style-Fused Diffusion Models</strong>（2026；医学影像、分割与稠密预测、息肉；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2026/0557-Paper4818.html">来源</a>；<a href="https://github.com/zhaiying0925/SFD-">代码/项目</a>）</li>
<li id="ref-2039"><span class="ref-no">[79]</span> <strong>Dual-Stream Diffusion with Structure-Aware Fusion for Joint Synthesis of Coherent Medical Image-Mask Pairs</strong>（2026；医学影像、分割与稠密预测、图像—标注联合生成；全文PDF已读取；<a href="https://doi.org/10.1007/978-3-032-38189-7_18">来源</a>）</li>
<li id="ref-2272"><span class="ref-no">[80]</span> <strong>Learning to Segment Burns with Limited Labels: A Dual-Path Cascaded Network with Uncertainty-Guided Teacher Diffusion Exposure</strong>（2026；医学影像、分割与稠密预测、标签稀缺与半监督；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2026/0573-Paper5539.html">来源</a>；<a href="https://github.com/VEDAs-Lab/EBIS.">代码/项目</a>）</li>
<li id="ref-2274"><span class="ref-no">[81]</span> <strong>Polyp-LDM: Accurate Boundary Alignment and Realism Enhancement for Colonoscopic Polyp Image-Mask Pair Generation</strong>（2025；医学影像、分割与稠密预测、图像—标注联合生成；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2025/0037-Paper2379.html">来源</a>；<a href="https://github.com/16rq/Polyp-LDM.">代码/项目</a>）</li>
<li id="ref-2035"><span class="ref-no">[82]</span> <strong>DiDGen: Diffusion-Based Dual-Task Synthesis for Dermoscopic Data Generation</strong>（2025；医学影像、分割与稠密预测、分类、双任务生成；全文PDF已读取；<a href="https://doi.org/10.1007/978-3-032-05141-7_8">来源</a>；<a href="https://github.com/junjie-shentu/DiDGen.">代码/项目</a>）</li>
<li id="ref-2276"><span class="ref-no">[83]</span> <strong>DiffAtlas: GenAI-fying Atlas Segmentation via Image-Mask Diffusion</strong>（2025；医学影像、分割与稠密预测、图谱/atlas；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2025/0231-Paper0228.html">来源</a>；<a href="https://github.com/M3DV/DiffAtlas.">代码/项目</a>；<a href="https://github.com/google-deepmind/surface-">代码/项目</a>）</li>
<li id="ref-240"><span class="ref-no">[84]</span> <strong>Diffusion-Based Data Augmentation for Medical Image Segmentation</strong>（2025；医学影像、分割与稠密预测；全文PDF已读取；<a href="https://doi.org/10.1109/iccvw69036.2025.00143">来源</a>）</li>
<li id="ref-2279"><span class="ref-no">[85]</span> <strong>Diffusion-Based User-Guided Data Augmentation for Coronary Stenosis Detection</strong>（2025；医学影像、冠脉检测、分类；全文PDF已读取；<a href="https://papers.miccai.org/miccai-2025/0238-Paper3613.html">来源</a>；<a href="https://github.com/medipixel/DiGDA.">代码/项目</a>）</li>
<li id="ref-2281"><span class="ref-no">[86]</span> <strong>LoFT: LoRA-Fused Training Dataset Generation with Few-shot Guidance</strong>（2025；分类、细粒度、少样本与数据稀缺；全文PDF已读取；<a href="https://arxiv.org/abs/2505.11703">来源</a>；<a href="https://github.com/ExplainableML/LoFT.">代码/项目</a>）</li>
<li id="ref-2283"><span class="ref-no">[87]</span> <strong>S2-RL: Sample-Set Dual Reinforcement Learning for Generative Semantic Segmentation Dataset Distillation</strong>（2026；分割与稠密预测、数据集蒸馏；全文PDF已读取；<a href="https://arxiv.org/abs/2610.35491">来源</a>；<a href="https://github.com/00why00/S2RL">代码/项目</a>；<a href="https://github.com/huggingface/diffusers">代码/项目</a>；<a href="https://github.com/open-mmlab/mmsegmentation">代码/项目</a>）</li>
<li id="ref-2285"><span class="ref-no">[88]</span> <strong>DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery</strong>（2026；数据集蒸馏、分类；全文PDF已读取；<a href="https://proceedings.mlr.press/v306/xia26g.html">来源</a>）</li>
<li id="ref-2287"><span class="ref-no">[89]</span> <strong>Efficient Dataset Distillation for Pre-Trained Self-Supervised Models via Statistical Flow Matching</strong>（2026；数据集蒸馏、表征学习；全文PDF已读取；<a href="https://arxiv.org/abs/2602.05391">来源</a>；<a href="https://github.com/einsteinxia/SFM.">代码/项目</a>）</li>
</ul>
</details>

<script>
(function () {
  function openReference() {
    var id = window.location.hash.slice(1);
    var target = id ? document.getElementById(id) : null;
    var box = target && target.closest('details');
    if (box) box.open = true;
  }
  window.addEventListener('hashchange', openReference);
  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[href^="#ref-"]');
    if (link) window.setTimeout(openReference, 0);
  });
  window.addEventListener('DOMContentLoaded', openReference);
})();
</script>
