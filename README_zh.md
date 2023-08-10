# ABAKA AI Awesome LLM

大语言模型（LLM）已经取得了举世瞩目的成就，每个人都有自己 AI 助手🤖的世界距离人类越来越近了。这个仓库整理了许多的大语言模型，我们将这些模型按照不同的训练方式划分了表格。

本仓库由整数智能团队组织创建，非常欢迎各界的小伙伴完善仓库内容。

[English](README.md) | 简体中文



# 目录

* [ABAKA AI Awesome LLM](#abaka-ai-awesome-llm)
  * [目录](#目录)
  * [更新内容](#更新内容)
  * [基础模型](#基础模型)
  * [监督微调模型](#监督微调模型)
  * [人类反馈强化学习模型](#人类反馈强化学习模型)
  * [灵感](#灵感)
  * [大语言模型扩展](#大语言模型扩展)

# 更新内容

* [2023-08-07]  第一天创建！！！🎉🎉🎉

# 基础模型

我们从基础模型开始总结，表格中的 AVG 代表从 [open-llm-leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 中获得的模型的平均得分。

|        模型        |    参数量    | 发布时间 |                         来源 & 资源                          |                 AVG                  |   Tokens数量    |  语言   |                             证书                             |
| :----------------: | :----------: | :------: | :----------------------------------------------------------: | :----------------------------------: | :-------------: | :-----: | :----------------------------------------------------------: |
| Switch Transformer |    1.6 T     | 2021-01  |    [paper](https://arxiv.org/pdf/2101.03961.pdf) \| none     |                                      |      5.3 B      | En most |                                                              |
|        GLaM        |    1.2 T     | 2021-12  |    [paper](https://arxiv.org/pdf/2112.06905.pdf) \| none     |                                      |      1.6 T      | En most |                                                              |
|        PaLM        |    540 B     | 2022-04  |    [paper](https://arxiv.org/pdf/2204.02311.pdf) \| none     |                                      |      780 B      | En most |                                                              |
|       MT-NLG       |    530 B     | 2022-01  |    [paper](https://arxiv.org/pdf/2201.11990.pdf) \| none     |                                      |      338 B      | En most |                                                              |
|      J1-Jumbo      |    178 B     | 2021-08  | [paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) \| none |                                      |      300 B      | En most |                                                              |
|        OPT         |    175 B     | 2022-05  | [paper](https://arxiv.org/pdf/2205.01068.pdf) \| [ckpt](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) |                                      |      180 B      | En most | [OPT-175B](https://github.com/facebookresearch/metaseq/blob/edefd4a00c24197486a3989abe28ca4eb3881e59/projects/OPT/MODEL_LICENSE.md) |
|       BLOOM        |    176 B     | 2022-11  | [paper](https://arxiv.org/pdf/2211.05100.pdf) \| [ckpt](https://huggingface.co/bigscience/bloom) |                                      |      350 B      | En most | [ License](https://huggingface.co/spaces/bigscience/license) |
|      GPT 3.0       |    175 B     | 2020-05  |    [paper](https://arxiv.org/pdf/2005.14165.pdf) \| none     |                                      |      499 B      | En most |                                                              |
|       LaMDA        |    137 B     | 2022-01  |    [paper](https://arxiv.org/pdf/2201.08239.pdf) \| none     |                                      |     2.81 T      | En most |                                                              |
|        GLM         |    130 B     | 2022-10  | [paper](https://arxiv.org/pdf/2210.02414.pdf) \| [ckpt](https://github.com/THUDM/GLM-130B) |                                      |      400 B      | En most | [GLM-130B](https://github.com/THUDM/GLM-130B/blob/799837802264eb9577eb9ae12cd4bad0f355d7d6/MODEL_LICENSE) |
|        YaLM        |    100 B     | 2022-06  | [blog](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6) \| [ckpt](https://github.com/yandex/YaLM-100B#downloading-checkpoint) |                                      | 1.7 TB(storage) | En & Ru | [Apache 2.0](https://github.com/yandex/YaLM-100B/blob/14fa94df2ebbbd1864b81f13978f2bf4af270fcb/LICENSE) |
|       LLaMA        | 7/13/33/65 B | 2022-09  | [paper](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) \| [ckpt](https://github.com/facebookresearch/llama#download) | 7B: 49.7<br/>33B: 61.7<br/>65B: 62.1 |      1.4 T      | En most | [Apacch 2.0](https://github.com/yandex/YaLM-100B/blob/main/LICENSE) |
|      GPT-NeoX      |     20 B     | 2022-04  | [paper](https://arxiv.org/pdf/2204.06745.pdf) \| [ckpt](https://github.com/EleutherAI/gpt-neox#pretrained-models) |                 46.4                 |     825 GiB     | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|       Falcon       |     40 B     | 2023-05  | [homepage](https://falconllm.tii.ae/) \| [ckpt](https://huggingface.co/tiiuae/falcon-40b) |                 61.5                 |       1 T       |         | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|        UL2         |     20 B     | 2022-05  | [paper](https://arxiv.org/pdf/2205.05131v1.pdf) \| [ckpt](https://huggingface.co/google/ul2) |                                      |      32 B       | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|     鹏程.盘古α     |     13 B     | 2021-04  | [paper](https://arxiv.org/pdf/2104.12369.pdf) \| [ckpt](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α#模型下载) |                                      |     26.5 B      | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|         T5         |     11 B     | 2019-10  | [paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) \| [ckpt](https://huggingface.co/t5-11b) |                                      |      34 B       | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|      CPM-Bee       |     10 B     | 2022-10  |    [paper](https://arxiv.org/pdf/2012.00413.pdf) \| none     |                                      |                 | Ch & En | [CPM-Bee](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md) |
|       rwkv-4       |     7 B      | 2022-09  | [paper](https://arxiv.org/pdf/2305.13048.pdf) \| [ckpt](https://huggingface.co/BlinkDL/rwkv-4-pile-7b) |                                      |                 |         |                                                              |
|       GPT-J        |     6 B      | 2022-09  | none \| [ckpt](https://github.com/kingoflolz/mesh-transformer-jax#pretrained-models) |                 42.8                 |      400 B      | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|      GPT-Neo       |    2.7 B     | 2021-03  | none \| [ckpt](https://github.com/EleutherAI/gpt-neo#pretrained-models) |                 38.9                 |                 |         | [MIT](https://github.com/EleutherAI/gpt-neo/blob/23485e3c7940560b3b4cb12e0016012f14d03fc7/LICENSE) |
|    baichuan-7B     |     7 B      | 2023-06  | [github](https://github.com/baichuan-inc/baichuan-7B) \| [ckpt](https://huggingface.co/baichuan-inc/Baichuan-7B) |                                      |      1.2 T      |         |                                                              |



# 监督微调模型

|      模型      |    参数量    | 发布时间 |                         来源 & 资源                          |                AVG                 | Tokens数量  |  语言   |                             证书                             |
| :------------: | :----------: | :------: | :----------------------------------------------------------: | :--------------------------------: | :---------: | :-----: | :----------------------------------------------------------: |
|    StableLM    | 3/7/15/65 B  | 2023-04  | [homepage](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) \| none |       3B: 34.3<br/>7B: 38.7        |    1.5 T    | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|   Flan-PaLM    |    540 B     | 2022-10  |    [paper](https://arxiv.org/pdf/2210.11416.pdf) \| none     |                                    |    1.4 B    |         |                                                              |
|     BLOOMZ     |    176 B     | 2022-11  | [paper](https://arxiv.org/pdf/2211.01786.pdf) \| [ckpt](https://huggingface.co/bigscience/bloomz) |                                    |   2.09 B    | En most | [License](https://huggingface.co/spaces/bigscience/license)  |
|  InstructGPT   |    175 B     | 2022-03  |    [paper](https://arxiv.org/pdf/2203.02155.pdf) \| none     |                                    |  13 K item  | En most |                                                              |
|   Galactica    |    120 B     | 2022-11  | [paper](https://arxiv.org/pdf/2211.09085.pdf) \| [ckpt](https://huggingface.co/facebook/galactica-120b) |                                    |    106 B    | En most |                                                              |
|  OpenChatKit   |     20 B     |  2023-3  | [github](https://github.com/togethercomputer/OpenChatKit) \| [ckpt](https://huggingface.co/togethercomputer/Pythia-Chat-Base-7B) |                                    |     1 T     | En most | [Apache 2.0](https://github.com/EleutherAI/gpt-neox/blob/main/LICENSE) |
|    Flan-UL2    |     20 B     | 2023-03  | [homepage](https://www.yitay.net/blog/flan-ul2-20b) \| [ckpt](https://huggingface.co/google/flan-ul2) |                49.1                |             |         |    [ Apache 2.0](https://huggingface.co/google/flan-ul2)     |
|     Gopher     |    280 B     |          | [paper](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) \| none |                                    |    300 B    |         |                                                              |
|   Chinchilla   |     70 B     |          |    [paper](https://arxiv.org/pdf/2203.15556.pdf) \| none     |                                    |    1.4 T    |         |                                                              |
|    Flan-T5     |     11 B     | 2022-10  | [paper](https://arxiv.org/pdf/2210.11416.pdf) \| [ckpt](https://huggingface.co/google/flan-t5-base) |                                    |    100 B    |         | [Apache 2.0](https://github.com/google-research/t5x/blob/776279bdacd8c5a2d3e8ce0f2e7064bd98e98b47/LICENSE) |
|       T0       |     11 B     | 2021-10  | [paper](https://arxiv.org/pdf/2110.08207.pdf) \| [ckpt](https://huggingface.co/bigscience/T0) |                                    |     1 T     |         |      [Apache 2.0](https://huggingface.co/bigscience/T0)      |
|     Alpaca     |     7 B      | 2023-03  | [github](https://github.com/tatsu-lab/stanford_alpaca) \| none |                31.9                |  52 K item  |         | [Apache 2.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) |
|      Orca      |     13 B     | 2023-06  |    [paper](https://arxiv.org/pdf/2306.02707.pdf) \| none     |                57.7                |  5 M item   |         | [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) |
|   ChatGLM-6B   |     6 B      | 2023-03  | [github](https://github.com/THUDM/ChatGLM-6B) \| [ckpt](https://huggingface.co/THUDM/chatglm-6b) |                                    |     1 T     | Ch & En | [Apache 2.0](https://github.com/THUDM/ChatGLM-6B/blob/main/LICENSE) |
|  ChatGLM2-6B   |     6 B      | 2023-06  | [github](https://github.com/THUDM/ChatGLM2-6B) \| [ckpt](https://huggingface.co/THUDM/chatglm2-6b) |                48.2                |    1.4 T    | Ch & En |  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)   |
|      Ziya      |     13 B     | 2023-05  | [github](https://github.com/IDEA-CCNL/Fengshenbang-LM) \| [ckpt](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1) |                32.3                |    125 B    | Ch & En | [Apache 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE) |
|    Phonenix    |     7 B      | 2023-04  | [github](https://github.com/FreedomIntelligence/LLMZoo) \| [ckpt](https://github.com/FreedomIntelligence/LLMZoo#phoenix-llm-across-languages) |                                    | 922 M item  | Ch & En | [Apache 2.0](https://github.com/FreedomIntelligence/LLMZoo/blob/main/LICENSE) |
|    Dolly2.0    |     12 B     | 2023-04  | [blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) \| [ckpt](https://huggingface.co/databricks/dolly-v2-12b) |                43.7                |  15 K item  |   En    | [Apache 2.0](https://github.com/databrickslabs/dolly/blob/master/LICENSE) |
|     Dolly      |     6 B      | 2023-03  | [github](https://github.com/databrickslabs/dolly) \| [ckpt](https://huggingface.co/databricks/dolly-v1-6b) |                                    |  52 K item  |   En    | [Apache 2.0](https://github.com/databrickslabs/dolly/blob/master/LICENSE) |
|    GALPACA     |     30 B     | 2022-11  | [paper](https://galactica.org/static/paper.pdf) \| [ckpt](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b) |                48.2                |  52 K item  |   En    | [CC By NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) |
|    UltraLM     |     13 B     | 2023-03  | [github](https://github.com/thunlp/UltraChat) \| [ckpt](https://huggingface.co/openbmb/UltraLM-13b) |                60.3                | 774 K item  |   En    | [CC By NC 4.0](https://github.com/thunlp/UltraChat/blob/main/LICENSE) |
|    Guanaco     | 7/13/33/65 B | 2023-03  | [github](https://github.com/artidoro/qlora) \| [ckpt](https://huggingface.co/timdettmersc) | 7B: 56.2<br/>13B: 59.1<br/>33B: 63 | 9.85 K item |   En    | [MIT License](https://github.com/artidoro/qlora/blob/main/LICENSE) |
|    BayLing     |     13 B     | 2023-06  | [github](https://github.com/ictnlp/BayLing) \| [ckpt](https://huggingface.co/ICTNLP/bayling-13b-v1.1) |                                    | 160 K item  | Ch & En | [GPL:v3](https://github.com/ictnlp/BayLing/blob/main/LICENSE) |
|     KnowLM     |     13 B     | 2023-06  | [github](https://github.com/zjunlp/KnowLM/) \| [ckpt](https://github.com/zjunlp/KnowLM/#2-2) |                                    | 1400 K item | Ch & En | [Apache 2.0](https://github.com/zjunlp/KnowLM/blob/main/LICENSE) |
|    WizardLM    |   13/30 B    | 2023-06  | [github](https://github.com/nlpxucan/WizardLM) \| [ckpt](https://huggingface.co/WizardLM) |      13B: 60.4<br/>30B: 62.9       | 143 K item  |   En    | [Apache 2.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) |
|     BELLE      |     7 B      | 2023-04  | [github](https://github.com/LianjiaTech/BELLE) \| [ckpt](https://huggingface.co/BelleGroup/BELLE-7B-2M) |                                    |  10 M item  |         | [Apache 2.0](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE) |
|     Koala      |     13 B     | 2023-04  | [homepage](https://bair.berkeley.edu/blog/2023/04/03/koala/) \| [ckpt](https://huggingface.co/TheBloke/koala-13B-HF) |                56.5                | 420 K item  |   En    |                                                              |
| Chinese-Vicuna |    7/13 B    | 2023-03  | [github](https://github.com/Facico/Chinese-Vicuna) \| [ckpt](https://huggingface.co/Facico) |                                    | 0.5 M item  | Ch & En | [Apache 2.0](https://github.com/Facico/Chinese-Vicuna/blob/master/LICENSE) |
|     Baize      |  7/13/30 B   | 2023-03  | [github](https://github.com/project-baize/baize-chatbot) \| [ckpt](https://huggingface.co/project-baize) |       7B: 51.3<br/>13B: 58.4       | 100 K item  | Ch & En | [License](https://github.com/project-baize/baize-chatbot/blob/main/LICENSE) |
|      MOSS      |     7 B      | 2023-04  | [github](https://github.com/OpenLMLab/MOSS) \| [ckpt](https://huggingface.co/fnlp/moss-moon-003-sft) |                                    |             |         | [LICENSE](https://github.com/OpenLMLab/MOSS/blob/main/MODEL_LICENSE) |



# 人类反馈强化学习模型

|    模型    | 参数量 | 发布日期 |                           来源                            |  Tokens数量  |
| :--------: | :----: | :------: | :-------------------------------------------------------: | :----------: |
|   GPT 4    | 1.8 T  | 2023-03  |         [blog](https://openai.com/research/gpt-4)         |     13 T     |
|  ChatGPT   | 175 B  |          |                                                           |              |
|  Sparrow   |  70 B  |          |                                                           |              |
|   Claude   |        |          |                                                           |              |
| StackLLaMA |  7 B   |          | [hugging face](https://huggingface.co/blog/zh/stackllama) | 10.8 M items |



# 灵感

我们整理了一些杰出的论文，并且总结了这些论文中提到的创新点或者解决了哪些存在的问题。

| 模型               | 论文链接                                                     | 灵感                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Switch Transformer | [paper](https://arxiv.org/pdf/2101.03961.pdf)                | MoE混合专家模型参数数量极高，但受到复杂性、通信成本和训练不稳定性的限制；本论文引入了Switch Transformer来解决这些问题，简化了MoE路由算法，设计了直观改进的模型，降低了通信和计算成本，并首次证明了可以使用较低精度（bfloat16）格式训练大型稀疏模型的可行性。 |
| GLaM               | [paper](https://arxiv.org/pdf/2112.06905.pdf)                | 训练大型而密集的模型需要大量的计算资源，该文章提出了一系列用于GLaM的语言模型，它使用稀疏激活的专家混合架构来扩展模型容量。最大的GLaM模型拥有1.2万亿个参数，大约是GPT-3大小的7倍。它仅消耗GPT-3训练能量的1/3，并且推理过程所需的计算量减少了一半，但在29个自然语言处理任务中仍然取得了更好的零、一和少量样本训练的综合性能。 |
| PaLM               | [paper](https://arxiv.org/pdf/2204.02311.pdf)                | 该文章进一步探讨了规模对少样本学习的影响，并通过在数百个语言理解和生成基准测试中实现最先进的少样本学习结果，展示了扩展规模的持续优势。 |
| MT-NLG             | [paper](https://arxiv.org/pdf/2201.11990.pdf)                | 该文章详细介绍了拥有5300亿个参数的转换器（Converter）为基础的最大单体语言模型Megatron-Turing NLG 530B (MT-NLG)的训练细节。 |
| J1-Jumbo           | [paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) | 该文章详细描述了AI21 Labs的J1-Jumbo（一个拥有1780亿个参数的参数模型）和J1-Large（一个拥有70亿个参数的参数模型），重点关注它们的架构和训练，并评估它们相对于GPT-3的性能。评估包括复杂性以及零样本学习和少样本学习的能力。 |
| OPT                | [paper](https://arxiv.org/pdf/2205.01068.pdf)                | 该文章介绍了开放预训练变换器（OPT），一组仅用于解码器的预训练变换器，参数范围从1.25亿到1.75亿不等。该文章旨在以全面负责的方式与感兴趣的研究人员分享这些变换器，并展示OPT-175B与GPT-31相媲美，其开发所需的碳足迹仅为GPT-31的七分之一。 |
| BLOOM              | [paper](https://arxiv.org/pdf/2211.05100.pdf)                | 大多数大型语言模型是由资源丰富的组织开发的，通常不对公众开放。为了推动这项强大技术的民主化进程，该文章介绍了BLOOM，一个由数百名研究人员合作设计和构建的拥有1760亿参数的开放式语言模型。BLOOM在各种基准测试中取得了竞争性的表现，并且在使用多任务提示进行微调时表现更加强大。 |
| GPT 3.0            | [paper](https://arxiv.org/pdf/2005.14165.pdf)                | 该文章指出了在任务特定微调方面取得的巨大进展，但仍然需要成千上万个实例的任务特定微调数据集。因此，该文章表明扩展语言模型可以显著提高在独立于任务的少实例情况下的性能，有时甚至可以与以往最先进的微调方法相媲美。 |
| LaMDA              | [paper](https://arxiv.org/pdf/2201.08239.pdf)                | 该文章证明，使用带注释的数据进行微调，并使模型能够参考外部知识来源，可以显著改善安全性和事实依据这两个关键挑战。安全性包括确保模型的回答与一系列人类价值观保持一致，例如防止有害建议和不公平偏见。事实依据包括使模型能够查阅外部知识来源，如信息检索系统、语言翻译器和计算器。 |
| GLM                | [paper](https://arxiv.org/pdf/2210.02414.pdf)                | 该文章主要介绍了GLM-130B的训练过程，包括其设计选择、为提高效率和稳定性而采取的训练策略以及工程工作。该文章试图开源一个至少与GPT-3一样好的1000亿规模模型，并揭示了如何成功地预训练这样大小的模型。 |
| LLaMA              | [paper](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) | 专有数据集难以获取，该论文介绍了LLaMA，一个包含从7B到65B参数的基础语言模型集合。研究人员在数万亿个标记上训练LLaMA的模型，并展示了只使用公开可用数据集而不依赖专有和不可访问的数据集，也可以训练出最先进的模型。 |
| GPT-NeoX           | [paper](https://arxiv.org/pdf/2204.06745.pdf)                | 该文章的主要贡献是将权重文件开源。                           |
| UL2                | [paper](https://arxiv.org/pdf/2205.05131v1.pdf)              | 该论文提出了一个统一的预训练模型框架，该框架在不同数据集和设置下通常非常有效。研究人员首先将架构原型与预训练目标分开，这两个概念经常被混淆。接下来，我们提出了自然语言处理中自监督学习的一般性和统一性观点，并展示了如何将不同的预训练目标互相转换和在不同目标之间插值的有效性。然后，我们提出了混合去噪器（MoD），这是一种结合了不同预训练范式的预训练目标。此外，引入了模式切换的概念，其中下游微调与特定的预训练方案相关联。 |
| 鹏程.盘古α         | [paper](https://arxiv.org/pdf/2104.12369.pdf)                | 该论文展示了训练一种名为PanGu-α的大规模自回归语言模型的实践，该模型拥有高达2000亿个参数。研究人员在文本摘要、问答和对话生成等各种场景中经验性地测试了PanGu-α的生成能力。此外，还研究了模型规模对中文自然语言处理任务中少样本性能的影响。 |
| T5                 | [paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf)  | 在这篇论文中，研究人员通过引入一个统一的框架，将所有基于文本的自然语言处理问题转化为文本到文本的格式，来探索NLP的迁移学习技术的领域。 |
| CPM-Bee            | [paper](https://arxiv.org/pdf/2012.00413.pdf)                | 将GPT-3应用于解决中文自然语言处理任务仍然具有挑战性，因为GPT-3的训练语料主要是英文，而且参数不公开。在这份技术报告中，研究人员发布了一个用于大规模中文训练数据的生成式预训练中文语言模型（CPM）。CPM拥有26亿个参数和100GB的中文训练数据，是目前最大的中文预训练语言模型，可以应用于多个下游中文自然语言处理任务，例如对话、文本生成、接近性测试和语言理解。广泛的实验证明，在少样本学习（甚至零样本学习）的设置下，CPM在许多自然语言处理任务上取得了出色的性能。 |
| rwkv-4             | [paper](https://arxiv.org/pdf/2305.13048.pdf)                | 与Transformer相比，循环神经网络（RNN）在内存和计算需求方面呈线性缩放，但由于并行化和可扩展性的限制，很难达到与Transformer相同的性能。该文章提出了一种新颖的模型架构，称为Reception Weighted Key-Value（RWKV），它将Transformer的高效并行训练与RNN的高效推断相结合。实验证明，RWKV的性能与相似规模的Transformer相当，这表明未来的工作可以利用这种架构创建更高效的模型。 |

| Model             | Link                                                         | Insight                                                      |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Flan-PaLM/Flan-T5 | [paper](https://arxiv.org/pdf/2210.11416.pdf)                | 该论文探索了指令微调（instruction fine-tuning）的方法，特别关注了以下三个方面：（1）任务数量的扩展，（2）模型规模的扩展，（3）基于思维链数据的微调。在上述方面进行指令微调显著提高了多个模型类别（PaLM、T5、U-PaLM）、设置（零样本、少样本、思维链）和评估基准（MMLU、BBH、TyDiQA、MGSM、开放式生成、RealToxicityPrompts）上的性能。 |
| BLOOMZ            | [paper](https://arxiv.org/pdf/2211.01786.pdf)                | 该论文发现，在英语任务上对大型多语言语言模型进行微调，并使用英语提示，能够实现对仅出现在预训练语料中的非英语语言的任务泛化。 |
| InstructGPT       | [paper](https://arxiv.org/pdf/2203.02155.pdf)                | 大型语言模型可能会生成不现实、有毒或对用户无益的输出。该论文通过基于人类反馈对语言模型进行微调，展示了在各种任务上将语言模型与用户意图对齐的路径。这种微调的方法可以帮助改善语言模型的输出，使其更符合用户的预期。 |
| Galactica         | [paper](https://arxiv.org/pdf/2211.09085.pdf)                | 该论文介绍了Galactica，一个能够存储、组合和推理科学知识的大型语言模型。研究人员使用大量的科学论文、参考资料、知识库和其他信息源进行训练。 |
| Gopher            | [paper](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) | 该文章分析了基于Transformer的语言模型在不同规模下的性能，从拥有数千万参数的模型到名为Gopher的拥有2800亿参数的模型。这些模型在152个不同的任务上进行了评估，并在其中大多数任务上取得了最先进的性能。 |
| Chinchilla        | [paper](https://arxiv.org/pdf/2203.15556.pdf)                | 该论文发现，当前大规模语言模型的训练存在明显的欠训练问题，这是由于最近将重点放在扩大语言模型规模上，同时保持训练数据量不变。为了实现计算上的最佳训练，模型大小和训练标记数量应该成等比例增长：每当模型大小翻倍时，训练标记的数量也应该翻倍。 |
| T0                | [paper](https://arxiv.org/pdf/2110.08207.pdf)                | 最近已经证明，大型语言模型在各种任务上可以实现合理的零样本泛化。为了测试零样本泛化是否可以直接通过显式的多任务学习来诱导，该论文开发了一个系统，可以将任何自然语言任务轻松转化为易于阅读的提示形式。 |
| Orca              | [paper](https://arxiv.org/pdf/2306.02707.pdf)                | 最近的研究集中在通过模仿学习来增强较小模型的能力，借鉴了大型基础模型（LFMs）生成的输出。这些模型的质量受到许多问题的影响，包括来自浅层LFM输出的有限模仿信号、小规模的同质训练数据，以及最显著的是缺乏严格的评估，导致高估小模型的能力，因为它们倾向于学习模仿LFM的风格，而不是推理过程。Orca通过学习模仿LFM的推理过程来解决这些挑战。 |
| UltraLM           | [paper](about:blank)                                         | 该论文旨在进一步提高开源模型的上限。首先，他们提供了一个经过系统设计、多样化、信息丰富的大规模指导性对话数据集UltraChat，该数据集不涉及人类查询。他们的目标是捕捉人类与AI助手之间可能发生的各种互动，并采用全面的框架迭代地生成多轮对话。 |
| Guanaco           | [paper](https://arxiv.org/pdf/2305.14314.pdf)                | 该论文介绍了QLORA，一种高效的微调方法，可以在单个48GB GPU上减少内存使用量，从而使得可以对一个拥有650亿参数的模型进行微调，同时保持完整的16位微调任务性能。 |
| Baize             | [paper](https://arxiv.org/pdf/2304.01196.pdf)                | 该论文提出了一种流程，通过利用ChatGPT与自身进行对话，可以自动生成高质量的多轮对话语料库。随后，他们使用参数高效微调来增强开源的大型语言模型LLaMA。所得到的模型名为Baize，在多轮对话中展现出良好的性能，并通过设置安全限制来最小化潜在风险。 |



# 大语言模型扩展

为了让大语言模型变得更加有趣，我们整理了一些基于大语言模型的有趣扩展，非常值得一试。

* [MindsDB](https://github.com/mindsdb/mindsdb): 将大语言模型与数据库的创建和操作相结合的项目
  * MindsDB是一个用于人工智能逻辑的服务器，使开发人员能够以快速且可扩展的方式将基于AI的系统从原型设计和实验推广到生产环境中。
* [ai](https://github.com/vercel-labs/ai): LLM 与代码生成相结合，这个项目可以借助AI构建React、Svelte、Vue和Solid应用
  * The Vercel AI SDK是一个用于构建基于人工智能的流式文本和聊天界面的库。它提供了一组工具和功能，帮助开发人员在应用程序中集成自然语言处理和聊天机器人功能，从而创建出强大的、实时的文本和聊天用户界面。
* [poerful-llms](https://github.com/howl-anderson/unlocking-the-power-of-llms): 使用prompts和chains让chatgpt称为神奇的生产力工具 
  * 使用 ChatGPT （也将包含 Google Bard）完成各种 NLP 任务和一些非 NLP 任务，这里面包括文本生成，润色等等。
* [engshell](https://github.com/emcf/engshell):  一款借助LLM来玩转各种操作系统的shell命令
  * 由LLMs（大型语言模型）支持的中文语言Shell是指一个命令行界面或Shell，利用大型语言模型的能力提供中文自然语言处理和理解功能。这个Shell可以在任何操作系统上使用，允许用户通过中文命令和查询与计算机进行交互。它利用LLMs的能力来理解和执行用户的指令、执行任务、提供信息，并可能根据用户输入的上下文提供智能建议或回应。
* [Superagent](https://github.com/homanp/superagent): 构建、部署和管理 LLM 支持的代理
  * Superagent是一个强大的工具，简化了大型语言模型（LLM）代理的配置和部署到生产环境的过程。它提供了一系列功能和功能，使开发人员更容易构建、管理和部署AI代理到生产环境中，包括内置内存和通过向量数据库进行文档检索的功能、强大的工具、Webhook、定时任务等等。Superagent的目标是提供一个全面而易于使用的平台，帮助开发人员更轻松地管理和部署LLM代理到生产环境中。
* [SillyTavern](https://github.com/SillyTavern/SillyTavern): 面向高级用户的 LLM 前端
  * 手机友好，多API（KoboldAI/CPP、Horde、NovelAI、Ooba、OpenAI+代理、WindowAI（Claude！）），类似视觉小说的Waifu模式，Horde SD，系统TTS，WorldInfo（传说书籍），可定制的用户界面，自动翻译，以及比您想要或需要的更多提示选项。可选的额外服务器提供更多的SD/TTS选项+ChromaDB/Summarize。
* [flux](https://github.com/paradigmxyz/flux): 基于图形的 LLM 动力工具，用于并行探索许多完成的项目
  * Flux是与大型语言模型（LLMs）交互的强大工具，它以树形结构生成多个完成选项，并允许您并行地探索最佳选项。
* [code-review-gpt](https://github.com/mattzcarey/code-review-gpt): 借助 LLM 模型的代码审查器
  * Code Review GPT是使用大型语言模型来审查您的CI/CD流水线中的代码。它通过提供反馈意见，帮助简化代码审查过程，指出可能存在问题或需要改进的代码部分。
* [vim-ai](https://github.com/madox2/vim-ai): Vim 的人工智能代码助手。适用于 Vim 和 Neovim 的 OpenAI 和 ChatGPT 插件
  * 这个插件为Vim和Neovim添加了人工智能（AI）功能。您可以使用OpenAI的API生成代码、编辑文本，或与GPT模型进行交互式对话。
* [readme-ai](https://github.com/eli64s/readme-ai): 从终端生成漂亮的 README.md 文件。由 OpenAI 的 GPT LLM 提供支持
  * README-AI是一个功能强大、用户友好的命令行工具，为您的软件和数据项目生成详尽的README markdown文档。通过提供远程仓库URL或代码库的目录路径，该工具将记录您整个项目的信息，利用大型语言模型和OpenAI的GPT API的能力。
* [mindflow](https://github.com/mindflowai/mindflow): AI 支持的 CLI git 包装器、样板代码生成器、聊天历史记录管理器和代码搜索引擎，可简化您的开发工作流程
  * 这是针对现代开发者的ChatGPT驱动的瑞士军刀！我们提供一个基于人工智能的命令行界面（CLI）的git封装器，代码样板生成器，代码搜索引擎，对话历史管理器等等。还有更多功能！
* [LangChain](https://github.com/langchain-ai/langchain): 通过可组合性使用 LLM 构建应用程序
  * 大型语言模型（LLMs）作为一项具有变革性的技术正在兴起，使开发人员能够构建以往无法实现的应用程序。然而，仅仅使用这些LLMs往往不足以创建一个真正强大的应用程序——真正的威力在于当您将它们与其他计算或知识来源相结合时。通过将LLMs与其他计算资源或知识源相结合，可以进一步拓展应用程序的功能和能力。这种组合可能包括结合其他AI模型、外部API、数据库或其他工具，以充分利用LLMs的潜力，并为应用程序提供更广泛的计算和知识支持。
* [haystack](https://github.com/deepset-ai/haystack): Haystack 是一个开源 NLP 框架，可使用 Transformer 模型和 LLM（GPT-4、Falcon 等）与数据进行交互
  * Haystack是一个端到端的自然语言处理（NLP）框架，使您能够构建由LLMs、Transformer模型、向量搜索等驱动的NLP应用程序。无论您想进行问答、答案生成、语义文档搜索，还是构建能够进行复杂决策和查询解析的工具，您都可以使用Haystack中的先进NLP模型构建解决您特定用例的端到端NLP应用程序。Haystack提供了一套工具和功能，使您能够轻松地使用这些模型来处理文本数据，进行信息检索和语义理解，并构建高性能的NLP应用程序。
* [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP): 一款简单易用且功能强大的自然语言处理开发库
* [postgresml](https://github.com/postgresml/postgresml): 一个人工智能应用数据库
  * PostgresML是一个针对PostgreSQL的机器学习扩展，使您能够使用SQL查询对文本和表格数据进行训练和推理。借助PostgresML，您可以将机器学习模型无缝集成到您的PostgreSQL数据库中，并利用先进的算法强大地处理数据。通过PostgresML，您可以使用SQL查询来训练和推理数据，并利用PostgreSQL数据库的功能和性能，以高效地执行机器学习任务。
* [anything-llm](https://github.com/Mintplex-Labs/anything-llm): 一个全栈应用程序，可将任何文档转变为智能聊天机器人，具有时尚的 UI 和更轻松的工作空间管理方式
  * 这是一个全栈应用程序和工具套件，可以将任何文档、资源或内容转化为数据片段，供任何LLM在对话过程中作为参考使用。该应用程序以非常低的开销运行，默认情况下LLM和vectorDB都是远程托管的，但也可以切换为本地实例。目前，该项目支持Pinecone、ChromaDB等用于向量存储，以及OpenAI用于LLM对话。
* [500+ best AI tools](https://vaulted-polonium-23c.notion.site/500-Best-AI-Tools-e954b36bf688404ababf74a13f98d126): 收集了不限于 LLM 模型的众多 AI 生产力工具