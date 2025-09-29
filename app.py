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
# These will be populated when the model is trained or loaded
MODEL_FEATURES = [] 
MODEL_DTYPES = None

# --- 1. Model Training Function ---
def train_and_save_model():
    """
    Loads data, trains the RandomForest model, captures feature info, and saves it.
    """
    global MODEL_FEATURES, MODEL_DTYPES
    print("--- Starting Model Training ---")
    
    # Load the dataset
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the dataset '{INPUT_CSV_FILE}'.")
        exit()

    # Safely convert date column
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    except Exception as e:
        print(f"FATAL ERROR: Could not parse dates in '{INPUT_CSV_FILE}'. Error: {e}")
        exit()

    # Feature Engineering
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    print("Feature engineering complete.")

    # Define Features (X) and Target (y)
    target = 'risk_level'
    features_to_drop = [target, 'date', 'village_name', 'bacterial_test_result']
    MODEL_FEATURES = [col for col in df.columns if col not in features_to_drop]

    X = df[MODEL_FEATURES]
    y = df[target]

    # *** NEW: Capture the data types of the training data ***
    MODEL_DTYPES = X.dtypes.to_dict()

    print(f"Features for training: {MODEL_FEATURES}")

    # Train the model
    print("Training the Random Forest model on the full dataset...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X, y)
    print("Model training complete.")

    # Save the trained model and features
    joblib.dump({'model': model, 'features': MODEL_FEATURES, 'dtypes': MODEL_DTYPES}, MODEL_FILE_PATH)
    print(f"Model and metadata saved successfully to '{MODEL_FILE_PATH}'")
    
    return model

# --- 2. Flask API Setup ---
app = Flask(__name__)
CORS(app) 

# Load the model and its metadata once when the server starts
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
        
        # Create a pandas DataFrame from the input data, ensuring correct column order
        input_df = pd.DataFrame([data], columns=MODEL_FEATURES)

        # *** FIX: Ensure data types match the training data before prediction ***
        input_df = input_df.astype(MODEL_DTYPES)

        # Make prediction and get probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

        # Prepare the response
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
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

