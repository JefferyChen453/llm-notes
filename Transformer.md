<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240507132314853.png" alt="image-20240507132314853" style="zoom:50%;" />

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240507132340703.png" alt="image-20240507132340703" style="zoom:50%;" />

### 如何理解QKV？

- **键（K）**：键的主要功能是帮助确定注意力分布，即模型应该“关注”输入序列中的哪些部分。在自注意力机制中，每个查询（Q）都会与所有的键（K）进行比较，通过计算它们之间的点积来评估相似度，从而形成一个注意力分数。这个分数决定了对应值（V）的加权重要性。
- **值（V）**：每个值包含了输入序列中某部分的信息内容，注意力机制通过使用由查询和键计算出的注意力分数来加权这些值，生成加权的输出，这个输出随后会被用于进一步的处理或生成最终的输出。

### 绝对位置编码

$PE_{pos, 2i} = sin(\frac{pos}{10000^{2i/d}})$

$PE_{pos, 2i} = cos(\frac{pos}{10000^{2i/d}})$

### 缩放点积除以 $\sqrt{d}$ 的原因

降低 qk 内积的方差，对 softmax 函数，值太大会导致梯度过小，不利于训练稳定。

### Q K 为什么要使用不同矩阵进行线性投影变换？

- 若 Q K 相同，则矩阵乘积结果是对称矩阵，减弱了表达能力
- 对角线元素值过大（因为相似度最高），每个位置过分关注自己
- 增大参数量，增强表征能力

### 为什么主流 LLM 都用 Decoder-Only？

- 双向注意力容易在训练中退化成低秩矩阵，而 Decoder-Only 使用下三角，维持满秩，建模能力更强
- Causal Attention 天然具有位置编码的功能
- 单向模型支持 KV Cache，对话场景效率高

### Pre-Norm 和 Post-Norm

- 原始 Transformer 用的是 post-norm，残差后进行归一化（add&norm），对参数正则化效果更强；利于构造更深的网络；对每个通路都进行了归一化，容易梯度消失。**强调残差分支**
- pre-norm 中残差部分不经过归一化通路，训练稳定，能够训练更深的网络；加大网络宽度而非深度 **强调 Identity 分支**

### Attention 时间复杂度计算

已知：$[m, n] * [n, p]$ 的计算复杂度为 $2*m*n*p$

- 第一部分：Attention
  - 计算 $Q、K、V$：$[N,d]*[d,d]->[N,d]$
    - $3*(2*N*d*d)=6Nd^2$
  - 计算 $Q*K^T$：$[N,d]*[d,N]->[N,N]$
    - $2*N*d*N=2N^2d$
  - 计算 $score*V$：$[N,N]*[N.d]->[N,d]$
    - $2*N*N*d=2N^2d$
  - 计算 $W_o*context$：$[N,d]*[d,d]->[N,d]$
    - $2*N*d*d=2Nd^2$
- 第二部分：MLP
  - 计算两个线性投影： $[N,d]*[d,4d]*[4d*d]->[N,d]$
    - $2*N*d*4d+2*N*4d*d=16Nd^2$

总计：$24Nd^2+4dN^2$

### Flash Attention

### Longformer