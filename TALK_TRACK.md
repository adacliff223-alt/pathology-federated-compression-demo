# 汇报讲稿

## 30 秒版本

老师您好，我做了一个面向多中心病理 AI 的联邦学习通信压缩 Demo。它模拟 3 家医院，各医院数据不出本地，只上传模型更新。我的重点不是单纯跑 FedAvg，而是比较了 Top-K 稀疏、SignSGD 量化和 error feedback 残差补偿。当前 CPU 实验中，FedAvg 的 AUC 是 0.9937，而 Top-K + SignSGD + error feedback 在梯度值载荷只有 0.312% 的情况下，AUC 仍能达到 0.9878。这个结果说明通信压缩如果配合残差补偿，有潜力用于多中心病理模型训练。

## 2 分钟版本

我关注的是 OmniPT 这类多中心病理 AI 项目里的协作训练问题。病理 WSI 数据体量大，而且涉及隐私和合规，实际中很难把多家医院的数据直接集中。

所以我先搭了一个联邦学习原型：服务器维护全局模型，3 家模拟医院各自在本地训练，然后只上传模型更新，服务器做 FedAvg 聚合。

在这个基础上，我进一步关注通信瓶颈。普通 FedAvg 每轮要上传完整更新，通信量是 100%。我加入了两个压缩策略：

第一是 Top-K 稀疏化，只传幅值最大的 10% 更新。

第二是 SignSGD，把 32-bit 浮点梯度压成正负方向，也就是 1-bit 信息。这样 Top-K + SignSGD 的梯度值载荷大约是 0.312%。

但实验里也能看到，单纯 Top-K + SignSGD 会让 AUC 崩掉，这说明极端压缩不能直接用。于是我加了 error feedback，也就是客户端把本轮没传出去的残差保存下来，下一轮继续补偿。

加入 error feedback 后，Top-K + SignSGD + EF 在 0.312% 通信载荷下，AUC 可以接近 FedAvg。这是我觉得这个 Demo 比较有意义的地方：它不只是把网页跑起来，而是展示了压缩通信里的一个真实问题和一个有效补偿方法。

目前这个版本是 CPU 友好的原型，默认用合成 PCam-like patch。下一步我计划换成真实 PatchCamelyon，加更严格的 Non-IID 划分，并统计索引和协议开销，让通信量统计更接近真实系统。

## 老师可能问的问题

### 这是真实病理数据吗？

当前默认不是，它是合成 PCam-like patch，用来保证 CPU 上可以快速复现实验流程。代码里保留了 `torchvision.datasets.PCAM` 入口，下一步可以替换为真实 PatchCamelyon。

### 为什么不用真实 WSI？

WSI 需要切 patch、组织区域筛选、存储管理和更大算力。这个 Demo 先验证联邦训练和通信压缩机制，后续可以接入真实 patch pipeline。

### 0.312% 是怎么算的？

Top-K 只保留 10% 更新，SignSGD 把每个 32-bit 浮点值变成 1-bit 符号，所以梯度值载荷比例是 `10% * 1 / 32 = 0.3125%`。

### 有没有计算索引开销？

当前主结果强调梯度值载荷，还没有把 Top-K 索引编码、通信协议和安全聚合开销完整加入。这个是后续要补的真实系统统计。

### 为什么单纯 Top-K + SignSGD 效果差？

因为它丢掉了大量幅值和非 Top-K 位置信息，更新方向过于粗糙。加入 error feedback 后，被丢弃的残差会在后续轮次继续补偿，所以训练稳定性明显提升。

### 没 GPU 能继续做吗？

能。CPU 可以继续做小模型、压缩算法、Non-IID 划分和通信统计。GPU 主要用于真实 PCam、ResNet18 或更大规模实验。

## 项目包装标题

可以用下面这个标题：

```text
面向多中心病理 AI 的联邦学习通信压缩 Demo：Top-K、SignSGD 与 Error Feedback
```

## GitHub 简介

```text
A CPU-friendly federated learning demo for pathology image classification, simulating three hospitals and evaluating gradient communication compression with Top-K sparsification, SignSGD quantization, and error feedback.
```
