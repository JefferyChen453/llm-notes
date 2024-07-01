### Adapter  Tuning

<img src="https://pic3.zhimg.com/80/v2-e61faf735e930fbf1c925a40385790de_1440w.webp" alt="img" style="zoom: 33%;" />

### Prefix Tuning

源码解读：https://zhuanlan.zhihu.com/p/635162651

<img src="https://pic3.zhimg.com/80/v2-f8697c52540aeed5c507753625c30202_1440w.webp" alt="img" style="zoom: 33%;" />

动机

- prompt 时，采用离散化的提示难以搜索到较好的提示
- 介于 ICL 和 全量微调之间，减少参数量

实现

- 将 prefix 加到每一层 transformer block 的 k、v 矩阵之前。假设30个virtual tokens，共 24 layers，n_dim=1024，则要训练的 prompt encoder 参数矩阵（或称为专为 30 个 virtual tokens 设计的 embedding layer 的 shape 为 [30, 24 * 1024 * 2]。每个 $P_\theta$ shape 为[30, 1024]`注意：prefix 不加入当前 prompt 中，否则就是 prompt tuning 了`

- 为了稳定训练，不会直接更新 $P_\theta$，而是多加一层 MLP [<1024, 1024]

实验

- 效果上，离散的提示 < embedding only < prefix tuning 这样的每一层都加前缀

### Prompt Tuning

源码解读：https://zhuanlan.zhihu.com/p/635233092

<img src="https://picx.zhimg.com/v2-61980563685b7b2029a436686d755b70_720w.jpg?source=d16d100b" alt="[代码学习]Huggingface的peft库学习-part 2- prompt tuning" style="zoom: 67%;" />

动机

- 在 prompt 的前面直接追加一定数量的 virtual tokens，然后专门对这些新增加的虚拟tokens进行微调

实现

- 直接在 prompt 前面加入可训练 tokens。模型设置跟上个保持不变，则新增的可训练参数为 [8, 1024]

### P-Tuning

![img](https://pic2.zhimg.com/v2-a7cfac58768e743ee6cb66c781297621_b.jpg)

- P-tuning 是一种自动构建模板的大模型微调技术。它将包含 prompt、context 和 target 的 template 中的 prompt，从人工设计的方式，转化成了优化连续参数的问题。具体的实现方式是利用 lstm 和 relu 构建了MLP，通过模型对伪token的embedding进行学习

- P-tuning v2 采用了 Prefix-tuning 的方式，微调了每一层 transformer的输入 embedding，在不同参数量级（330M~10B）模型上均取得了不错的效果，并且可以完成复杂的下游任务（序列标注）。