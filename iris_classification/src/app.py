from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

model = joblib.load('models/logistic_regression.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1,-1)
    
        prediction = model.predict(features)[0]
        return jsonify({
            'input':data['features'],
            'prediction' : prediction
        })
    except Exception as e:
        return jsonify({"error":str(e)})
    
if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)