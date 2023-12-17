from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    distribution_channel = request.form['channel']
    first_month_sales = float(request.form['first_month_sales'])
    second_month_sales = float(request.form['second_month_sales'])

    # Make prediction
    prediction = model.predict([[first_month_sales, second_month_sales]])

    # Return results
    return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
