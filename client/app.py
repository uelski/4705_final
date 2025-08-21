import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
# touch .env
# echo 'API_URL="http://<FASTAPI-EC2-PUBLIC-IP>:8000/predict' > .env"

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

    st.caption("Select any applicable toxicity types:")
    cols = st.columns(3)
    checkbox_state = {}
    for i, label in enumerate(TOXICITY_LABELS):
        with cols[i % 3]:
            pretty = label.replace("_", " ").title()
            checkbox_state[label] = st.checkbox(pretty, key=f"cb_{label}")

    submitted = st.form_submit_button("Submit")

# handle submit
if submitted:
    if not comment or not comment.strip():
        st.error("Please enter a non-empty comment.")
        st.stop()

    selected_labels = [lbl for lbl, val in checkbox_state.items() if val]
    
    true_labels = {t: int(t in selected_labels) for t in TOXICITY_LABELS}

    payload = {
        "text": comment.strip(),
        "true_labels": true_labels
    }

    # send to api here and show result
    res = requests.post(API_URL, json=payload)
    output = res.json()

    if output:
        st.header('Model prediction:')
        st.subheader("Original Text:")
        st.text(comment)
        for key, data in output.items():
            if key == "response":
                for sentiment,value in data.items():
                    st.subheader(f"{sentiment}")
                    st.text(f"Predicted: {value} | True: {true_labels[sentiment]}")
                    st.divider()

    else:
        st.header('Error processing predictions.')
        st.error("Please try again later.", icon="🚨")
