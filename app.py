from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model pipeline
MODEL_PATH = "best_poverty_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please run the training script first.")

model = joblib.load(MODEL_PATH)

# HTML Template with modern design (Internal for simplicity in this script)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poverty Headcount Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0d9488;
            --primary-dark: #0f766e;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #2dd4bf;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(to right, #2dd4bf, #0d9488);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        .form-section {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        label {
            font-weight: 500;
            color: var(--accent);
            display: flex;
            justify-content: space-between;
        }

        label span {
            color: var(--text-muted);
            font-weight: 400;
        }

        input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
            height: 6px;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            cursor: pointer;
        }

        .prediction-section {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 16px;
            padding: 30px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            border: 1px dashed rgba(45, 212, 191, 0.3);
            transition: all 0.3s ease;
        }

        .prediction-value {
            font-size: 4rem;
            font-weight: 700;
            color: var(--accent);
            margin: 20px 0;
            text-shadow: 0 0 20px rgba(45, 212, 191, 0.4);
        }

        .prediction-label {
            font-size: 1.2rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .info-card {
            margin-top: 40px;
            background: rgba(45, 212, 191, 0.05);
            padding: 20px;
            border-radius: 12px;
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.6;
            border-left: 4px solid var(--accent);
        }

        .btn-predict {
            background: var(--primary);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
        }

        .btn-predict:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SDG 1: No Poverty</h1>
            <p class="subtitle">Poverty Headcount Predictor Dashboard</p>
        </header>

        <div class="main-content">
            <div class="form-section">
                <div class="input-group">
                    <label for="household_size">Average Household Size <span id="val_household">4.5</span></label>
                    <input type="range" id="household_size" min="1" max="10" step="0.1" value="4.5">
                </div>

                <div class="input-group">
                    <label for="electricity_access">Electricity Access (%) <span id="val_electricity">95</span></label>
                    <input type="range" id="electricity_access" min="0" max="100" step="1" value="95">
                </div>

                <div class="input-group">
                    <label for="annual_income">Average Annual Income (₹) <span id="val_income">60000</span></label>
                    <input type="range" id="annual_income" min="5000" max="150000" step="1000" value="60000">
                </div>

                <button class="btn-predict" onclick="getPrediction()">Update Prediction</button>
            </div>

            <div class="prediction-section" id="prediction_box">
                <div class="prediction-label">Predicted Headcount Ratio</div>
                <div class="prediction-value" id="result">--</div>
                <div class="prediction-label">Percent (H)</div>
            </div>
        </div>

        <div class="info-card">
            <strong>Model Info:</strong> This prediction is powered by a Gradient Boosting Regressor trained on regional socioeconomic data. 
            The Headcount Ratio (H) represents the proportion of the population living below the poverty line.
        </div>
    </div>

    <script>
        const inputs = ['household_size', 'electricity_access', 'annual_income'];
        const displays = ['val_household', 'val_electricity', 'val_income'];

        inputs.forEach((id, index) => {
            const input = document.getElementById(id);
            input.addEventListener('input', (e) => {
                document.getElementById(displays[index]).innerText = e.target.value;
            });
        });

        async function getPrediction() {
            const data = {
                'Average Household Size': parseFloat(document.getElementById('household_size').value),
                'Electricity Access': parseFloat(document.getElementById('electricity_access').value),
                'Average Annual Income': parseFloat(document.getElementById('annual_income').value)
            };

            const resultDiv = document.getElementById('result');
            resultDiv.innerText = "...";

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                // Animate value
                let current = 0;
                const target = result.prediction;
                const duration = 500;
                const startTime = performance.now();

                function update(now) {
                    const elapsed = now - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const currentVal = (progress * target).toFixed(2);
                    resultDiv.innerText = currentVal;
                    
                    if (progress < 1) {
                        requestAnimationFrame(update);
                    }
                }
                requestAnimationFrame(update);

            } catch (error) {
                console.error('Error:', error);
                resultDiv.innerText = "Error";
            }
        }

        // Initial prediction
        getPrediction();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Create DataFrame for prediction (must match training features)
        input_df = pd.DataFrame([data])
        
        # Get prediction from pipeline (scaling is handled by the pipeline)
        prediction = model.predict(input_df)[0]
        
        # Ensure prediction is not negative (though model should be accurate)
        prediction = max(0, prediction)
        
        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Running on port 5000
    print("Starting Poverty Predictor Web App...")
    app.run(debug=True, port=5000)
