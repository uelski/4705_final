import streamlit as st
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from pathlib import Path
import boto3

# setup
TABLE_NAME = ""
AWS_REGION = ""
DDB_ENDPOINT = ""

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# show banner
alert_placeholder = st.empty()

@st.cache_data(show_spinner=False)
def get_table(table_name: str, region: str, endpoint: str):
    session = boto3.Session(region_name=region)
    if endpoint:
        ddb = session.resource("dynamodb", endpoint_url=endpoint)
    else:
        ddb = session.resource("dynamodb")
    return ddb.Table(table_name)

@st.cache_data(show_spinner=True)
def fetch_all_logs(table_name: str, region: str, endpoint: str, limit: int):
    """
    Scan to collect up to `limit` items.
    Expected item keys:
      - timestamp (str/datetime ISO)
      - request_text (str)
      - response (dict of label->0/1)
      - true_labels (dict of label->0/1)
    """
    table = get_table(table_name, region, endpoint)

    

# title and description
st.title('Toxic Comment Moderation Monitoring App')
st.markdown("This app will be used to monitor the backend FastAPI application by plotting different data to help in analysing model performance.")



# display metrics
# st.metric("Accuracy:", f"{accuracy:.2%}")
# st.metric("Precision:", f"{precision:.2%}")

# Implement Alerting: If the calculated accuracy drops below 80%, display a prominent warning banner at the top of the dashboard using st.error().
# if accuracy < 0.80:
#     alert_placeholder.error(f"Warning: Model accuracy dropped to {accuracy:.2%}!", icon="🚨")