import streamlit as st
import json
# import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
# from pathlib import Path
import boto3
# from decimal import Decimal

# setup
TABLE_NAME = os.getenv("DDB_TABLE", "table_01")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def _to_int_map(d: dict) -> dict:
    if not isinstance(d, dict):
        return {k: 0 for k in LABELS}
    return {k: int(d.get(k, 0)) for k in LABELS}

def fetch_all_logs():
    ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = ddb.Table(TABLE_NAME)

    resp = table.scan()
    items = resp.get("Items", [])

    # normalize to your expected shape
    out = []
    for it in items:
        out.append({
            "timestamp": it.get("timestamp"),
            "request_text": it.get("request_text", ""),
            "response": _to_int_map(it.get("response", {})),
            "true_labels": _to_int_map(it.get("true_labels", {})),
        })
    return out


# dummy json to test
def load_logs_from_file(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data

logs = fetch_all_logs()
# logs = load_logs_from_file('dummy_logs.json')

def build_prediction_df(logs):
    rows = []
    for log in logs:
        row = {}
        for label in LABELS:
            row[label] = log["response"].get(label)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

pred_df = build_prediction_df(logs)

pred_dist = pred_df.mean().rename("positive_rate").reset_index()
pred_dist.columns = ["label", "positive_rate"]

def build_true_df(logs):
    rows = []
    for log in logs:
        row = {}
        for label in LABELS:
            row[label] = log["true_labels"].get(label)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

true_df = build_true_df(logs)
true_dist = true_df.mean().rename("positive_rate").reset_index()
true_dist.columns = ["label", "positive_rate"]

# for accuracy and precision display

y_true = true_df[LABELS].to_numpy()
y_pred = pred_df[LABELS].to_numpy()

accuracies = {}
for label in LABELS:
    accuracies[label] = accuracy_score(true_df[label], pred_df[label])

exact_match_acc = accuracy_score(y_true, y_pred)

precisions = {}
for label in LABELS:
    precisions[label] = precision_score(true_df[label], pred_df[label], zero_division=0)
    
precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)

# show banner
alert_placeholder = st.empty()

# title and description
st.title('Toxic Comment Moderation Monitoring App')
st.markdown("This app will be used to monitor the backend FastAPI application by plotting different data to help in analysing model performance.")

st.subheader(f"Total Logs: {len(logs)}")
# prediction distribution plots
st.subheader("Distribution of 1s per label - Prediction")
st.dataframe(pred_dist)
st.bar_chart(pred_dist.set_index("label"))

# true distribution plots
st.subheader("Distribution of 1s per label - True Labels")
st.dataframe(true_dist)
st.bar_chart(true_dist.set_index("label"))

# display accuracy metrics
st.subheader("Accuracy Metrics")
st.metric("Exact Match Accuracy:", f"{exact_match_acc:.2%}")
# per label accuracy
st.text("Per Label Accuracy")
st.dataframe(pd.DataFrame.from_dict(accuracies, orient="index", columns=["accuracy"]))

# display precision metrics
st.subheader("Precision Metrics")
st.metric("Macro Average Precision:", f"{precision_macro:.2%}")
# per label precision
st.text("Per Label Precision")
st.dataframe(pd.DataFrame.from_dict(precisions, orient="index", columns=["precision"]))

# Implement Alerting: If the calculated accuracy drops below 80%, display a prominent warning banner at the top of the dashboard using st.error().
if exact_match_acc < 0.50:
    alert_placeholder.error(f"Warning: Model accuracy dropped to {exact_match_acc:.2%}!", icon="🚨")