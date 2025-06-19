from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo previamente entrenado
modelo = joblib.load('modelo_iris.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Crear DataFrame con los nombres EXACTOS usados al entrenar el modelo
        data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

        # Realizar la predicción
        prediction = modelo.predict(data)

        return jsonify({'clase': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
