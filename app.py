from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for


# Initialize the Flask application.
app = Flask(__name__)


# Define project paths for saved model artifacts.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "floods.save"
SCALER_PATH = PROJECT_ROOT / "models" / "transform.save"


# Feature names must match the exact order used during model training.
FEATURE_NAMES = [
    "Temp",
    "Humidity",
    "Cloud Cover",
    "ANNUAL",
    "Jan-Feb",
    "Mar-May",
    "Jun-Sep",
    "Oct-Dec",
    "avgjune",
    "sub",
]


def load_artifact(path):
    """Load a saved joblib artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return joblib.load(path)


# Load the trained model and scaler once during application startup.
try:
    model = load_artifact(MODEL_PATH)
    scaler = load_artifact(SCALER_PATH)
except Exception as error:
    model = None
    scaler = None
    startup_error = error
else:
    startup_error = None


@app.route("/")
def home():
    """Render the landing page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the flood prediction input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Read form inputs, scale them, and render the prediction result."""
    try:
        if startup_error is not None or model is None or scaler is None:
            raise RuntimeError(f"Model artifacts could not be loaded: {startup_error}")

        # Read and validate all feature values from the submitted form.
        input_values = {}
        for feature in FEATURE_NAMES:
            raw_value = request.form.get(feature)

            if raw_value is None or raw_value.strip() == "":
                raise ValueError(f"Missing value for feature: {feature}")

            input_values[feature] = float(raw_value)

        # Build a one-row DataFrame using the exact training feature names.
        input_data = pd.DataFrame([input_values], columns=FEATURE_NAMES)

        # Apply the saved StandardScaler before prediction.
        scaled_input = scaler.transform(input_data)

        # Generate the flood prediction.
        prediction = model.predict(scaled_input)[0]

        if int(prediction) == 1:
            return render_template("chance.html")

        return render_template("no_chance.html")

    except Exception as error:
        # Keep the user on the form page and provide a clear error message.
        return render_template("index.html", error=str(error)), 400


@app.route("/home")
def redirect_home():
    """Redirect /home requests to the main landing page."""
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
