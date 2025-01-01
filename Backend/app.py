from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models and transformations
log_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
pca = joblib.load('pca_transform.pkl')  # If PCA was applied during training
scaler = joblib.load('scaler.pkl')      # If scaling was applied

# Route to make predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON data
        data = request.json
        print("Received data:", data)  # Debug log

        # Extract features from the incoming data
        time = data.get('Time', None)
        amount = data.get('Amount', None)
        features = data.get('V', None)
        print(time, amount, features)  # Expecting a list for 'V1' to 'V28'

        # Check if all required features are provided
        if time is None or amount is None or features is None or len(features) != 28:
            return jsonify({'error': f'Missing required features: Time, Amount, or V1 to V28. Features length: {len(features) if features else "None"}'}), 400

        # Prepare the input feature array (Time, Amount, V1 to V28)
        raw_features = np.array([[time, amount] + features])

        # Scale the numerical features (Time, Amount, and V1 to V28)
        scaled_features = scaler.transform(raw_features)

        # Apply PCA transformation
        transformed_features = pca.transform(scaled_features)

        # Make predictions using each model
        log_prediction = log_model.predict(transformed_features)
        rf_prediction = rf_model.predict(transformed_features)
        gb_prediction = gb_model.predict(transformed_features)

        # Get the majority vote from the models (optional)
        prediction = np.round((log_prediction + rf_prediction + gb_prediction) / 3)

        # Determine the result message
        if prediction[0] == 0:
            message = "Transaction is legitimate."
        else:
            message = "Transaction is fraudulent. Please investigate further."

        # Return the prediction and message
        return jsonify({'prediction': int(prediction[0]), 'message': message})

    except Exception as e:
        # Handle unexpected errors gracefully
        return jsonify({'error': str(e)}), 500


# Run the Flask app on the correct port (dynamic port from Render)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
