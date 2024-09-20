
from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model_pipeline.pkl')

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        main_category = request.form['main_category']
        sub_category = request.form['sub_category']
        ratings = float(request.form['ratings'])
        no_of_ratings = int(request.form['no_of_ratings'])
        actual_price = float(request.form['actual_price'])
        
        # Create a DataFrame for model input
        input_data = pd.DataFrame({
            'main_category': [main_category],
            'sub_category': [sub_category],
            'ratings': [ratings],
            'no_of_ratings': [no_of_ratings],
            'actual_price': [actual_price]
        })

        # Make the prediction
        predicted_price = model.predict(input_data)[0]

        # Return the result
        return render_template('result.html', predicted_price=predicted_price)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
