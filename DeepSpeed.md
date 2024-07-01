# DeepSpeed

## 1	ZeRO

Stage 0: 不做分片，相当于 DDP

Stage 1: $P_{os}$ 把**优化器状态(optimizer states)** 分片到每个数据并行的工作进程(每个GPU)下，通信量与 Stage0 相同

Stage 2: $P_{os+g}$ 把**优化器状态(optimizer states) + 梯度(gradients)**分片到每个数据并行的工作进程(每个GPU)下，通信量与 Stage0 相同

Stage 3: $P_{os+g+p}$ 把**优化器状态(optimizer states) + 梯度(gradients) + 模型参数(parameters)**分片到每个数据并行的工作进程(每个GPU)下，通信量增加

`os: FP32-Parameters FP32-Variance FP32-Momentum`

`g/p: FP16`

eg：一个参数量 $\psi$ 的模型，FP16 存储，原始大小  $2\psi$ B；训练中要存储自身参数和梯度，共  $4\psi$ B；AMP 中，Adam 需要一份 FP32 的模型参数拷贝、一份 variance、一份 momentum，需 $3*4\psi$ B。共计  $16\psi$ B。

![img](https://pic1.zhimg.com/v2-66fd8b0d3b0f7dbfb491d148cf491540_b.jpg)

## 2	Stage-3 traininig iteration

1. **分布：**Data 分成 4 块，分别放到 4 个 GPU；

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418112536264.png" alt="image-20240418112536264" style="zoom:40%;" />

2. **前向传播：**GPU0 将 M0 参数广播到其他 GPU 上，每个 GPU 用自己的 Data 计算 activation（最上面色块） 和前向值，算完后 GPU1/2/3 上删除 M0；

   M1/2/3 同理，最终得到四个 Data 分别的 Loss；`M3 最后算好后 GPU0/1/2 上先不释放 M3`

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418112757894.png" alt="image-20240418112757894" style="zoom:40%;" />

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418121518836.png" alt="image-20240418121518836" style="zoom: 40%;" />

3. **反向传播：**从 M3 开始，每个 GPU 上分别计算 M3 上的 Gradient，如图送到 GPU3 执行 gradient accumulation，此时再将 GPU0/1/2 上的 M3 参数和激活值释放；

   然后 M2 --> M1 --> M0 依次执行；

   <img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418122148321.png" alt="image-20240418122148321" style="zoom: 40%;" />

   <img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418123141354.png" alt="image-20240418123141354" style="zoom:40%;" />

4. **优化器更新参数：**Optimizer 开始**并行**，计算出 FP32 的 updated weights，再转化为 FP16 weights，更新每块 GPU 上分布的参数，结束

<img src="C:\Users\Jeffery Chen\AppData\Roaming\Typora\typora-user-images\image-20240418123331009.png" alt="image-20240418123331009" style="zoom:40%;" />





## 2	Offload

将数据、梯度、优化器状态等下沉到CPU内存或硬盘上，节省显存。

![img](https://pic1.zhimg.com/v2-c476b637dda9652cfb90984b054a7420_b.jpg)

Offload就是把这个流程用上图的方式把 FP32 参数的更新和 float2half 拆分到了CPU和内存上计算，而前向和后向传播依然由GPU负责



## 3	Gradient Checkpoint

- 原理：反向传播时重新计算前向传播中的中间值，时间换空间
