## LLaMA 2

<img src="https://pic4.zhimg.com/80/v2-c9b10194c5e0aa9777afa984063e7ff3_720w.webp" alt="img" style="zoom: 67%;" />

### RMSNorm

LayerNorm：$y = \frac{x - E[x]}{\sqrt{Var[x]}+\epsilon}*\gamma+\beta$

RMSNorm：$y = \frac{x}{\sqrt{Mean(x^2)}}*\gamma$

### RoPE

对于 $q_m$ 和 $k_n$ ，其在 attention 相当于做内积，RoPE 的核心思想就是给 $q_m$ 和 $k_n$ 注入位置信息  $f()$ 变成 $\widetilde{q}_m$ 和 $\widetilde{k}_n$ ，使得内积 $<\widetilde{q}_m, \widetilde{k}_n>$ 中包含 $m - n$ ，这样就有了相对位置信息。

对于二维情况，![image-20240505234841612](C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240505234841612.png)

对于 n 维情况，![image-20240505234922760](C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240505234922760.png)

简化计算：![image-20240505234959141](C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240505234959141.png)

### KV Cache

![img](https://pic3.zhimg.com/80/v2-f764447457c75f18681e3f8bfdea20fe_720w.webp)

对于“生”，不用再重复计算一遍“生”之前的 token 对应的 K 和 V 了，直接从 cache 中读取。

为何不用缓存 Q？这是由单向注意力决定的，对于一个 past token，当前 token 的位置在其右侧，past tokens 不需要参与当前 token 的注意力计算，相当于每个 q 用完就丢。

### MQA & GQA

![img](https://pic1.zhimg.com/80/v2-0b4046dca50ceb80361ef1ee1ba3f6d4_720w.webp)

MQA (Multi Query Attention) ：Q 依然保持多头，但是 K、V 只有一个，所有多头的 Q 共享一个 K、V，这样做虽然能最大程度减少 KV Cache 所需的缓存空间，但是可想而知参数的减少意味着精度的下降，所以为了在精度和计算之间做一个 trade-off

GQA (Group Query Attention)：孕育而生，即 Q 依然是多头，但是分组共享 K、V，即减少了 K、V 缓存所需的缓存空间，也保留了大部分参数不至于精度损失严重