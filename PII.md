### 背景

这个比赛是一个 NER 任务，要求是找到文章中涉及到个人身份信息的 token，标签采用 BIO 格式，即 begin inside outside，其中共包含姓名、地址、电话等 7 种 PII 信息，每种信息对应 B 和 I 两种标签。

### 数据

用到了两个数据集

- train dataset：6600
- mixtral：2692 由 Mixtral8*7B 生成
- mistral：1350 由 Mistral 生成

官方给的数据集在标签平衡上不好，200 篇文章只有 O 标签，不含任何 PII 信息。补充数据集补充了更多标签。

### 训练

由于模型对 B- I- 的概念掌握不好，所以去除了标签中的 B- I-，只用 O+7个 PII 标签训练，在后处理中再加上 B- I-

learning rate = 1e-4 cosine 调度器

O 类别权重设为 0.2

- deberta：maxlen = 1024 accumulated bs = 16 4-fold train + mixtral
- deberta：maxlen = 2048 accumulated bs = 8 4-fold train + mistral
- deberta：maxlen = 3072 accumulated bs = 8 mixtral + mistral 冻结前六层 + lora (alpha = 8，dropout = 0.2) train

### 后处理

分词：将 deberta token 拆成 character，重新拼成 spacy token，prob 取 （最大或者取 mean 都试过，发现取最大效果更好）

过滤：将 prob 低于 0.9 的一律判断为非 PII

对特殊情况特殊判断：如对 NAME 标签中的 Mr. 等称谓标签设为 O

### 改进

超参数选取，可以考虑网格搜索