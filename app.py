from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os

app = Flask(_name_)

MODEL_PATH = "best_poverty_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = joblib.load(MODEL_PATH)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Poverty Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial; background:#0f172a; color:white; padding:30px; }
        .box { max-width:600px; margin:auto; background:#1e293b; padding:25px; border-radius:10px; }
        h2 { color:#2dd4bf; text-align:center; }
        input { width:100%; padding:6px; margin-top:5px; }
        button { width:100%; padding:10px; margin-top:15px; background:#0d9488; color:white; border:none; border-radius:6px; cursor:pointer; }
        .result { margin-top:20px; padding:15px; text-align:center; border-radius:8px; background:#111827; }
        .low { color:#22c55e; }
        .moderate { color:#facc15; }
        .high { color:#ef4444; }
    </style>
</head>
<body>
<div class="box">
    <h2>SDG 1 - Poverty Headcount Predictor</h2>

    <label>Average Household Size (1–10)</label>
    <input type="number" id="household" min="1" max="10" value="4.5" step="0.1">

    <label>Electricity Access (%) (0–100)</label>
    <input type="number" id="electricity" min="0" max="100" value="95">

    <label>Average Annual Income (₹)</label>
    <input type="number" id="income" min="1000" value="60000">

    <button onclick="predict()">Predict</button>

    <div class="result" id="output">Enter values and click Predict</div>
</div>

<script>
async function predict() {

    const output = document.getElementById("output");
    output.innerHTML = "Calculating...";

    const data = {
        "Average Household Size": parseFloat(document.getElementById("household").value),
        "Electricity Access": parseFloat(document.getElementById("electricity").value),
        "Average Annual Income": parseFloat(document.getElementById("income").value)
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.error) {
        output.innerHTML = "Error: " + result.error;
        return;
    }

    output.innerHTML = `
        <h3>${result.prediction}%</h3>
        <p class="${result.risk_class}">${result.risk_level}</p>
    `;
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validation
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        if not (1 <= data["Average Household Size"] <= 10):
            return jsonify({"error": "Household size must be between 1 and 10"}), 400

        if not (0 <= data["Electricity Access"] <= 100):
            return jsonify({"error": "Electricity access must be 0–100"}), 400

        input_df = pd.DataFrame([data])
        prediction = float(model.predict(input_df)[0])

        # Ensure prediction between 0–100%
        prediction = max(0, min(100, round(prediction, 2)))

        # Risk classification
        if prediction > 40:
            risk_level = "High Poverty Risk"
            risk_class = "high"
        elif prediction > 20:
            risk_level = "Moderate Poverty Risk"
            risk_class = "moderate"
        else:
            risk_level = "Low Poverty Risk"
            risk_class = "low"

        return jsonify({
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_class": risk_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if _name_ == "_main_":
    print("Poverty Predictor App Running...")
    app.run(port=5000)