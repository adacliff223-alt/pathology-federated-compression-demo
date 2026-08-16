# Pathology Federated Learning Demo

用公开病理数据集 PatchCamelyon 的训练流程，模拟 3 家医院在数据不出院的前提下做联邦学习；核心展示 Top-K 稀疏化 + SignSGD 量化如何把梯度值载荷从 100% 压到约 0.3%。

## 一句话

面向多中心病理 AI 协作场景，本 Demo 证明：医院只上传压缩梯度，也能完成联邦训练，并显著降低通信开销。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── compression.py
├── data.py
├── model.py
├── train_federated.py
├── make_figures.py        # 生成结果图
├── PROJECT_REPORT.md     # 正式项目报告
├── TALK_TRACK.md          # 套磁/汇报讲稿
├── RELEASE_CHECKLIST.md   # 项目包装顺序
├── figures/
│   └── auc_communication.svg
├── client.py
├── server.py
└── demo
    └── app.py
```

## 快速开始

```bash
pip install -r requirements.txt
python train_federated.py --rounds 3 --samples-per-hospital 600 --mode research --cpu
python -m streamlit run demo/app.py
```

默认使用轻量合成病理 patch，适合先跑通流程。要切换到 PatchCamelyon：

```bash
python train_federated.py --use-pcam --rounds 10 --samples-per-hospital 5000
```

## 核心函数

```python
def top_k_sparsify(vector, k_ratio=0.1):
    k = max(1, int(vector.numel() * k_ratio))
    values, indices = torch.topk(vector.abs(), k)
    sparse = torch.zeros_like(vector)
    sparse[indices] = vector[indices]
    return sparse

def sign_quantize(vector):
    return torch.sign(vector)
```

## 实验模式

| 模式 | 含义 | CPU 友好 |
|---|---|---|
| `fedavg` | 完整梯度通信基线 | 是 |
| `topk` | 只上传幅值最大的 10% 更新 | 是 |
| `topk_ef` | Top-K + error feedback 残差补偿 | 是 |
| `topk_sign` | Top-K + 1-bit 符号量化 | 是 |
| `topk_sign_ef` | Top-K + SignSGD + error feedback | 是 |

运行后会生成：

- `results.csv`：每轮 AUC 与通信量
- `summary.csv`：每种方法最后一轮结果，适合放进汇报

## 没有 GPU 怎么办

默认 `tiny` CNN 和合成 PCam-like patch 是专门为 CPU 准备的，600 张/医院、3 轮通常几十秒内能完成。没有 GPU 时不要直接上 `--model resnet18 --use-pcam --samples-per-hospital 5000`，那是后续 Colab 或实验室机器版本。

## 怎么包装这个项目

建议按下面顺序展示，而不是只打开网页：

1. 看 `PROJECT_REPORT.md`：说明临床问题、方法、实验结果和局限。
2. 再运行 `train_federated.py`：证明结果是可复现的，不只是静态页面。
3. 打开 `summary.csv`：展示每种方法最终 AUC 和通信量。
4. 最后打开 Streamlit 页面：作为可视化入口，而不是项目本体。

项目的核心贡献可以表述为：

```text
我不是只做了一个网页，而是搭了一个 CPU 可复现的联邦学习通信压缩实验闭环：
FedAvg 基线 -> Top-K 稀疏 -> SignSGD 量化 -> Error Feedback 残差补偿 -> AUC/通信量对比。
```

## 展示话术

基于病理 patch 数据搭建了一个模拟多医院联邦学习系统，比较了 FedAvg、Top-K 稀疏、SignSGD 量化和 error feedback 残差补偿。结果显示，在数据不出院的设定下，Top-K + SignSGD 能把梯度值载荷压缩到约 0.3%，error feedback 可用于缓解压缩带来的精度损失，这个方向可以服务 OmniPT 多中心协作里的通信优化问题。
