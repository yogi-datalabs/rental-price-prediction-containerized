from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).with_name("rental_price_prediction_model_v1_0.joblib")
FEATURES = [
    "accommodates",
    "bathrooms",
    "review_scores_rating",
    "bedrooms",
    "beds",
    "room_type",
    "cancellation_policy",
    "cleaning_fee",
    "instant_bookable",
]
NUMERIC_FEATURES = [
    "accommodates",
    "bathrooms",
    "review_scores_rating",
    "bedrooms",
    "beds",
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def to_boolean(value):
    """Convert common JSON/CSV boolean values to a Python bool."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError("cleaning_fee must be true or false")


def prepare_input(data):
    """Validate and normalize input before sending it to the model pipeline."""
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    prepared = data[FEATURES].copy()
    for feature in NUMERIC_FEATURES:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="raise")
    prepared["cleaning_fee"] = prepared["cleaning_fee"].map(to_boolean)
    prepared["instant_bookable"] = (
        prepared["instant_bookable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "t", "false": "f", "yes": "t", "no": "f"})
    )
    return prepared


def predict_prices(data):
    prepared = prepare_input(data)
    log_predictions = model.predict(prepared)
    return [round(float(value), 2) for value in np.exp(log_predictions)]


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/v1/rental")
def predict_single():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    try:
        prediction = predict_prices(pd.DataFrame([payload]))[0]
        return jsonify({"predicted_price_usd": float(prediction)})
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed"}), 500


@app.post("/v1/rentalbatch")
def predict_batch():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Upload a CSV file using the 'file' field"}), 400

    try:
        data = pd.read_csv(uploaded_file)
        predictions = predict_prices(data)
        identifiers = data["id"].tolist() if "id" in data.columns else data.index.tolist()
        results = [
            {"id": identifier, "predicted_price_usd": float(price)}
            for identifier, price in zip(identifiers, predictions)
        ]
        return jsonify({"count": len(results), "predictions": results})
    except (TypeError, ValueError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction could not be completed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
