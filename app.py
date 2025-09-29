import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
INPUT_CSV_FILE = 'health_data_final.csv'
MODEL_FILE_PATH = 'risk_model.pkl'
MODEL_FEATURES = [] 
MODEL_DTYPES = None

# --- 1. Model Training Function ---
def train_and_save_model():
    """
    Loads data, trains the RandomForest model, captures feature info, and saves it.
    """
    global MODEL_FEATURES, MODEL_DTYPES
    print("--- Starting Model Training ---")
    
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the dataset '{INPUT_CSV_FILE}'.")
        exit()

    try:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    except Exception as e:
        print(f"FATAL ERROR: Could not parse dates in '{INPUT_CSV_FILE}'. Error: {e}")
        exit()

    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    print("Feature engineering complete.")

    target = 'risk_level'
    features_to_drop = [target, 'date', 'village_name', 'bacterial_test_result']
    MODEL_FEATURES = [col for col in df.columns if col not in features_to_drop]

    X = df[MODEL_FEATURES]
    y = df[target]

    MODEL_DTYPES = X.dtypes.to_dict()

    print(f"Features for training: {MODEL_FEATURES}")

    print("Training the Random Forest model on the full dataset...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X, y)
    print("Model training complete.")

    joblib.dump({'model': model, 'features': MODEL_FEATURES, 'dtypes': MODEL_DTYPES}, MODEL_FILE_PATH)
    print(f"Model and metadata saved successfully to '{MODEL_FILE_PATH}'")
    
    return model

# --- 2. Flask API Setup ---
app = Flask(__name__)
CORS(app) 

if not os.path.exists(MODEL_FILE_PATH):
    model = train_and_save_model()
else:
    print(f"Loading existing model from '{MODEL_FILE_PATH}'...")
    saved_data = joblib.load(MODEL_FILE_PATH)
    model = saved_data['model']
    MODEL_FEATURES = saved_data['features']
    MODEL_DTYPES = saved_data['dtypes']
    print("Model and metadata loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    """ API endpoint to make predictions. """
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data], columns=MODEL_FEATURES)
        input_df = input_df.astype(MODEL_DTYPES)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

        response = {
            'prediction': int(prediction),
            'risk_label': risk_labels.get(prediction, 'Unknown'),
            'probabilities': {
                'low': probabilities[0],
                'medium': probabilities[1],
                'high': probabilities[2]
            }
        }
        return jsonify(response)
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return jsonify({'error': str(e)}), 500

# --- 3. Run the Flask App ---
if __name__ == '__main__':
    # Get the port from the environment variable Render sets, default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    # Run the app on 0.0.0.0 to make it accessible from outside the container
    app.run(host='0.0.0.0', port=port)

