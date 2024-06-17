from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
# Load the model
with open('cancer.pkl', 'rb') as file:
    model = pickle.load(file)

# Render the home page with input form
@app.route('/')
def home():
    return render_template('cancer_index.html')  # Use 'cancer_index.html' for the cancer predictor form

# Handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            Age = float(request.form['Age'])
            Gender = float(request.form['Gender'])
            BMI = float(request.form['BMI'])
            Smoking = float(request.form['Smoking'])
            GeneticRisk = float(request.form['GeneticRisk'])
            PhysicalActivity = float(request.form['PhysicalActivity'])
            AlcoholIntake = float(request.form['AlcoholIntake'])
            CancerHistory = float(request.form['CancerHistory'])

            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'Age': [Age],
                'Gender': [Gender],
                'BMI': [BMI],
                'Smoking': [Smoking],
                'GeneticRisk': [GeneticRisk],
                'PhysicalActivity': [PhysicalActivity],
                'AlcoholIntake': [AlcoholIntake],
                'CancerHistory': [CancerHistory]
            })

            # Use the model to make predictions
            prediction = model.predict(input_data)

            # Assuming 'Diagnosis' is the target variable
            diagnosis = prediction[0]

            return render_template('cancer_result.html', diagnosis=diagnosis)

        except Exception as e:
            error_message = str(e)
            return render_template('cancer_index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
