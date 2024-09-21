from flask import Flask, render_template, request
from model import predict_species
from sklearn.datasets import load_iris

app = Flask(__name__)
iris = load_iris()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    predicted_class = iris.target_names[prediction]
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
