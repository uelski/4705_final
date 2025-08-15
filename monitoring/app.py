import streamlit as st
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from pathlib import Path

# show banner
alert_placeholder = st.empty()

# title and description
st.title('Toxic Comment Moderation Monitoring App')
st.markdown("This app will be used to monitor the backend FastAPI application by plotting different data to help in analysing model performance.")



# display metrics
# st.metric("Accuracy:", f"{accuracy:.2%}")
# st.metric("Precision:", f"{precision:.2%}")

# Implement Alerting: If the calculated accuracy drops below 80%, display a prominent warning banner at the top of the dashboard using st.error().
# if accuracy < 0.80:
#     alert_placeholder.error(f"Warning: Model accuracy dropped to {accuracy:.2%}!", icon="🚨")