import streamlit as st
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

# title and description
st.title('Toxic Comment Moderation App')
st.markdown("This app is the frontend for our Toxic Comment Moderation ML project. Interact below by inputting text and selecting from the different toxicity options and submitting.")

TOXICITY_LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# form area
with st.form("tox_form", clear_on_submit=True):
    comment = st.text_area(
        "Comment", placeholder="Type a comment to analyze for toxicity...", height=140
    )

    st.caption("Select any applicable toxicity types (from training labels):")
    cols = st.columns(3)
    checkbox_state = {}
    for i, label in enumerate(TOXICITY_LABELS):
        with cols[i % 3]:
            pretty = label.replace("_", " ").title()
            checkbox_state[label] = st.checkbox(pretty, key=f"cb_{label}")

    submitted = st.form_submit_button("Analyze")

# handle submit
if submitted:
    if not comment or not comment.strip():
        st.error("Please enter a non-empty comment.")
        st.stop()

    selected_labels = [lbl for lbl, val in checkbox_state.items() if val]

    payload = {
        "text": comment.strip(),
        "true_labels": selected_labels
    }

    # send to api here and show result