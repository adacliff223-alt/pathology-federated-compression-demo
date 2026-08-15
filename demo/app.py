from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compression import estimate_communication
from model import get_model


st.set_page_config(page_title="Pathology FL Demo", layout="wide")

st.title("病理联邦学习通信压缩 Demo")

with st.sidebar:
    hospital = st.selectbox("Hospital", ["Hospital A", "Hospital B", "Hospital C"])
    k_ratio = st.slider("Top-K ratio", 0.01, 0.30, 0.10, 0.01)
    st.metric("染色风格", {"Hospital A": "bright / high contrast", "Hospital B": "mild stain", "Hospital C": "strong contrast"}[hospital])

model = get_model("tiny")
total_params = sum(parameter.numel() for parameter in model.parameters())
fedavg = estimate_communication(total_params, k_ratio=1.0, use_sign=False)
topk = estimate_communication(total_params, k_ratio=k_ratio, use_sign=False)
topk_sign = estimate_communication(total_params, k_ratio=k_ratio, use_sign=True)

left, mid, right = st.columns([1, 1, 1.2])

with left:
    st.subheader("医院数据")
    st.metric("本地样本", "600 patches")
    st.metric("数据策略", "数据不出院")
    st.progress(1.0, text="本地训练完成后仅上传梯度更新")

with mid:
    st.subheader("通信节省")
    st.metric("FedAvg", f"{fedavg.ratio * 100:.1f}%")
    st.metric("Top-K", f"{topk.ratio * 100:.1f}%")
    st.metric("Top-K + SignSGD", f"{topk_sign.ratio * 100:.2f}%")
    st.progress(topk_sign.saving, text=f"节省 {topk_sign.saving * 100:.2f}%")

with right:
    st.subheader("通信量对比")
    chart = pd.DataFrame(
        {
            "method": ["FedAvg", "Top-K", "Top-K + SignSGD"],
            "communication_%": [fedavg.ratio * 100, topk.ratio * 100, topk_sign.ratio * 100],
        }
    )
    st.bar_chart(chart, x="method", y="communication_%")

results_path = ROOT / "results.csv"
if results_path.exists():
    results = pd.read_csv(results_path)
    st.subheader("实验结果")
    st.line_chart(results, x="round", y="auc", color="mode")
    st.dataframe(results, width="stretch")

    summary_path = ROOT / "summary.csv"
    if summary_path.exists():
        st.subheader("最终轮汇总")
        st.dataframe(pd.read_csv(summary_path), width="stretch")
else:
    st.info("运行 python train_federated.py --mode research 后，这里会显示 AUC 曲线。")
