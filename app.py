import joblib
from flask import Flask, request, jsonify, render_template_string

# Load your trained model
model = joblib.load("tuned_injury_model.pkl")

# Number of features expected
EXPECTED_FEATURES = 21

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        if request.method == "POST":
            data = request.get_json()
            features = data.get("features")
        else:  # GET request with query parameters
            features_param = request.args.get("features")
            if features_param:
                features = [float(x) for x in features_param.split(",")]
            else:
                return jsonify({"error": "No features provided"}), 400

        if not features:
            return jsonify({"error": "No features provided"}), 400

        if len(features) != EXPECTED_FEATURES:
            return jsonify({"error": f"Expected {EXPECTED_FEATURES} features, but got {len(features)}."}), 400

        # Make prediction
        prediction = model.predict([features])[0]
        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([features])[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return {"message": "Injury Prediction API is running"}


@app.route("/form", methods=["GET", "POST"])
def form():
    html_form = """
        <h2>Injury Prediction Form</h2>
        <form method="post">
        {% set features = [
            'Training_Hours_Per_Week',
            'Stress_Level_Score',
            'Workload_Stress_Index',
            'Sleep_Efficiency',
            'Recovery_Score',
            'Injury_Risk_Score',
            'BMI',
            'Fitness_Workload_Ratio',
            'Recovery_Deficit',
            'session_load_mean',
            'session_load_std',
            'session_load_max',
            'ACWR_mean',
            'ACWR_max',
            'ACWR_last',
            'load_spike_sum',
            'monotony_mean',
            'strain_mean',
            'pitch_condition_mean',
            'conflict_level_mean',
            'match_intensity_sum'
        ] %}
        
        {% for f in features %}
            {{f}}: <input type="text" name="{{f}}"><br><br>
        {% endfor %}
        
        <input type="submit" value="Predict">
    </form>

    {% if prediction is defined %}
        <h3>Prediction: {{prediction}}</h3>
        {% if probability %}
            <p>Probabilities: {{probability}}</p>
        {% endif %}
    {% endif %}

    """

    if request.method == "POST":
        try:
            features = []
            for i in range(EXPECTED_FEATURES):
                value = request.form.get(f"f{i}")
                features.append(float(value))

            prediction = model.predict([features])[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba([features])[0].tolist()

            return render_template_string(html_form, expected=EXPECTED_FEATURES, prediction=int(prediction), probability=probability)
        except Exception as e:
            return render_template_string(html_form, expected=EXPECTED_FEATURES, prediction=f"Error: {str(e)}")

    return render_template_string(html_form, expected=EXPECTED_FEATURES)

from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    
    features = [
        'Training_Hours_Per_Week',
        'Stress_Level_Score',
        'Workload_Stress_Index',
        'Sleep_Efficiency',
        'Recovery_Score',
        'Injury_Risk_Score',
        'BMI',
        'Fitness_Workload_Ratio',
        'Recovery_Deficit',
        'session_load_mean',
        'session_load_std',
        'session_load_max',
        'ACWR_mean',
        'ACWR_max',
        'ACWR_last',
        'load_spike_sum',
        'monotony_mean',
        'strain_mean',
        'pitch_condition_mean',
        'conflict_level_mean',
        'match_intensity_sum'
    ]

    if request.method == "POST":
        try:
            # Convert all form inputs to float with safe fallback
            input_data = []
            for f in features:
                val = request.form.get(f, "").strip()  # get field
                if val == "":  
                    val = 0      # if blank, treat as 0
                input_data.append(float(val))

            # Convert to NumPy array in correct shape
            X = np.array(input_data).reshape(1, -1)

            # Example: use your trained model
            prediction = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(X).tolist()

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("form.html", prediction=prediction, probability=probability)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
