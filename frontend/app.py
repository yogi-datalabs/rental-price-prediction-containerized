import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

st.set_page_config(page_title="Airbnb Rental Price Prediction", page_icon="🏠")
st.title("🏠 Airbnb Rental Price Prediction")
st.caption("Streamlit frontend → Flask API → trained XGBoost model")

single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

with single_tab:
    with st.form("prediction_form"):
        room_type = st.selectbox(
            "Room type", ["Entire home/apt", "Private room", "Shared room"]
        )
        accommodates = st.number_input("Guests", min_value=1, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5, value=1.0)
        cancellation_policy = st.selectbox(
            "Cancellation policy", ["strict", "moderate", "flexible"]
        )
        cleaning_fee = st.checkbox("Cleaning fee charged", value=True)
        instant_bookable = st.checkbox("Instantly bookable")
        review_scores_rating = st.number_input(
            "Review score", min_value=0.0, max_value=100.0, value=90.0
        )
        bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
        beds = st.number_input("Beds", min_value=0, value=1)
        submitted = st.form_submit_button("Predict rental price", type="primary")

    if submitted:
        payload = {
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "cancellation_policy": cancellation_policy,
            "cleaning_fee": cleaning_fee,
            "instant_bookable": "t" if instant_bookable else "f",
            "review_scores_rating": review_scores_rating,
            "bedrooms": bedrooms,
            "beds": beds,
        }
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/rental", json=payload, timeout=30
            )
            response.raise_for_status()
            price = response.json()["predicted_price_usd"]
            st.success(f"Predicted rental price: **${price:,.2f}**")
        except requests.RequestException as error:
            message = "The prediction service is unavailable."
            if error.response is not None:
                message = error.response.json().get("error", message)
            st.error(message)

with batch_tab:
    st.write("Upload a CSV containing the same nine model features.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        preview = pd.read_csv(uploaded_file)
        st.dataframe(preview.head(), use_container_width=True)
        uploaded_file.seek(0)
        if st.button("Run batch prediction"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/rentalbatch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                response.raise_for_status()
                result = pd.DataFrame(response.json()["predictions"])
                st.success(f"Generated {len(result)} predictions.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    result.to_csv(index=False),
                    "rental_predictions.csv",
                    "text/csv",
                )
            except requests.RequestException as error:
                message = "The batch prediction service is unavailable."
                if error.response is not None:
                    message = error.response.json().get("error", message)
                st.error(message)

with st.sidebar:
    st.header("Container status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"Backend: {health['status']}")
    except (requests.RequestException, KeyError, ValueError):
        st.error("Backend: unavailable")
    st.code(BACKEND_URL, language=None)
