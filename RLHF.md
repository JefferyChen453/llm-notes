# RLHF （InstructGPT）

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418205525479.png" alt="image-20240418205525479" style="zoom:100%;" />

- **3 Datasets**
  - SFT dataset (13k)：prompt + 人工和 API 写的 response
  - RM dataset (33k)：模型根据 prompt 生成的 response 的**人工排序**
  - PPO dataset (31k)：仅有 prompts，由 SFT model 生成答案，由 RM 打分，进行 RLHF，无人工标注

- **SFT：** 微调 GPT-3 (175B)，扫了 16 epochs，SFT model 经过一个 epoch 就会过拟合，但实验发现 overfitting 有助于 RM score 和 human preference ratings
- **RM**： 训练 GPT (6B)，GPT 原本的最后一层是 softmax 输出概率。而现在把 softmax 拿掉，换成 linear_proj，输出维度为1，这个标量就是 score。`用小模型作为 RM 的原因：训练稳定`
  - **Loss：**由于输出是排序结果，要转换成值，故采用 Pairwise Ranking Loss
- **PPO：**$\pi^{RL}$ 是要学习的模型，其初始化就是 $\pi^{SFT}$。x 是来自 PPO Dataset 的 prompt，y 是 $\pi^{RL}$ 生成的 response
  - 第一项：目标是最大化 $r_\theta(x,y)$ 即 RM 的打分。
  - 第二项：$r_\theta$ 是用 $\pi^{SFT}$ 生成的 RM dataset 训练出来的，而此时的输入 y 是由在线学习的 $\pi^{RL}$ 生成的。数据分布不同，因此第二项是一个 KL 散度（衡量两个分布像不像）
  - 第三项：如果只有前两项，会导致 model 对对齐人类这个任务比较好，而在最初 pretrain 的 dataset 上性能下降。因此第三项是原始预训练的目标函数

![image-20240418211621224](C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418211621224.png)

`为什么要用 RL 来做？只用 step-1 中的 SFT 也可以，但需要非常大的人工成本来写 response；用 RL 就可以将打标任务转化为排序。`

- Actor 模型：$\pi^{RL}$ 由 SFT 初始化，不断产生 action，并被 Critic 模型打分

- Reference 模型：$\pi^{SFT}$ 约束 Actor 模型和 Reference 模型的 KL Penalty，防止 Actor 模型跑偏
- Reward 模型：$r_\theta$ step-2 中预训练，RLHF 中参数冻结
- Critic 模型：由 Reward 模型初始化，参数可训练，预测 Actor 模型生成 token

**采样指的是 old_policy 从 prompt 池中抽出 M 个 prompt 后，对每个prompt进行语言模型的token采样**：

- 计算response的第1个token的概率分布，然后从概率分布中采样出第1个token
- 根据第1个token，计算response的第2个token的概率分布，然后从概率分布中采样出第2个token
- ……
- 根据前N-1个token，计算response的第N个token的概率分布，然后从概率分布中采样出第N个token