from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the trained model
model_filename = 'heart_disease_model.pkl'
model = joblib.load(model_filename)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Define the route for prediction
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # If the request is a GET request (you used query parameters in your example)
    if request.method == 'GET':
        # Extract parameters from URL query string
        parameters = request.args
        features = [parameters.get('Age'), parameters.get('Gender'), parameters.get('Polyuria'), 
                    parameters.get('Polydipsia'), parameters.get('sudden_weight_loss'), parameters.get('weakness'),
                    parameters.get('Polyphagia'), parameters.get('Genital_thrush'), parameters.get('visual_blurring'),
                    parameters.get('Itching'), parameters.get('Irritability'), parameters.get('delayed_healing'),
                    parameters.get('partial_paresis'), parameters.get('muscle_stiffness'), parameters.get('Alopecia'),
                    parameters.get('Obesity')]
    else:
        # For POST requests, handle JSON data (used for Dialogflow)
        req = request.get_json(silent=True, force=True)
        if req is None:
            return jsonify({"fulfillmentText": "No data received"})
        # Extract parameters from Dialogflow's JSON request
        parameters = req.get('queryResult').get('parameters')
        features = [parameters['Age'], parameters['Gender'], parameters['Polyuria'], 
                    parameters['Polydipsia'], parameters['sudden_weight_loss'], parameters['weakness'],
                    parameters['Polyphagia'], parameters['Genital_thrush'], parameters['visual_blurring'],
                    parameters['Itching'], parameters['Irritability'], parameters['delayed_healing'],
                    parameters['partial_paresis'], parameters['muscle_stiffness'], parameters['Alopecia'],
                    parameters['Obesity']]
    
    # Convert features to numeric and reshape to match the model input
    features = np.array(features, dtype=float).reshape(1, -1)
    
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    # Return the result as a JSON response
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    
    # For Dialogflow (POST) response
    if request.method == 'POST':
        response = {
            "fulfillmentText": f"The prediction result is: {result}"
        }
        return jsonify(response)
    
    # For GET request response
    return jsonify({'prediction': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
