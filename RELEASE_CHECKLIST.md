# 项目包装顺序

## 1. 先发项目报告

优先发送：

```text
PROJECT_REPORT.md
```

这份文件负责说明：为什么做、用了什么方法、实验怎么设定、结果是什么、局限在哪里、下一步怎么做。

## 2. 再附最终结果表

附上：

```text
summary.csv
```

重点看最后一轮结果：

```text
FedAvg:                  AUC 0.9937, communication 100.000%
Top-K:                   AUC 0.9908, communication 10.000%
Top-K + SignSGD:         AUC 0.2416, communication 0.312%
Top-K + SignSGD + EF:    AUC 0.9878, communication 0.312%
```

核心结论：极端压缩会损伤训练，但 error feedback 可以显著补偿压缩误差。

## 3. GitHub 里放完整材料

建议 GitHub 首页保留这些文件：

```text
README.md
PROJECT_REPORT.md
TALK_TRACK.md
requirements.txt
compression.py
data.py
model.py
train_federated.py
demo/app.py
results.csv
summary.csv
figures/auc_communication.svg
make_figures.py
启动Demo.bat
```

README 负责快速说明和运行方式；PROJECT_REPORT 负责正式解释；TALK_TRACK 负责汇报话术；结果图负责让老师快速抓住结论。

## 4. Streamlit 网页最后展示

网页不是项目核心，只是可视化入口。启动方式：

```bat
启动Demo.bat
```

或者手动运行：

```bat
D:\.venv\Scripts\python.exe -m streamlit run demo\app.py
```

打开后访问：

```text
http://localhost:8501
```

## 5. 一句话包装

```text
我做了一个 CPU 可复现的多中心病理联邦学习通信压缩 Demo。它不是只展示网页，而是比较 FedAvg、Top-K、SignSGD 和 error feedback，在 AUC 与通信量之间建立可复现实验对比。
```
