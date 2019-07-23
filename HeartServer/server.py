import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from keras.models import load_model

app = Flask(__name__)

@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/api', methods=['POST'])
def predict():
    # Load the model
    model = load_model('models/full_model_190723.h5')

    # Fix the scaler
    train_values = 'input/train_values.csv'
    X = pd.read_csv(train_values)

    # Label Encoding
    X.thal = X['thal'].astype('category')
    X = pd.concat([X, pd.get_dummies(X.thal, 'thal')], axis=1)
    X = X.drop(['thal'], axis=1)

    # Scaling the data
    X = X.drop(['patient_id'], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Get the data from the POST request.
    print("trying to get json")
    data = request.get_json(force=True)
    print("got json from postman")
    ### Manipulate data
    testval = pd.DataFrame(data, index=[0])
    thal_value = testval.thal.iloc[0]
    testval['thal_reversible_defect'] = 0
    testval['thal_fixed_defect'] = 0
    testval['thal_normal'] = 0

    if thal_value == 'reversible_defect':
        testval['thal_reversible_defect'] = 1
    elif thal_value == 'normal':
        testval['thal_normal'] = 1
    elif thal_value == 'fixed_defect':
        testval['thal_fixed_defect'] = 1
    else:
        print("no valid thal value")

    testval = testval.drop(['thal'], axis=1)
    patient_id_test = testval.patient_id
    testval = testval.drop(['patient_id'], axis=1)
    ### End of manipulation

    # Make prediction using model loaded from disk as per the data.
    testval = scaler.transform(testval)
    testval = testval.reshape(-1, 15)
    prediction = model.predict(testval)

    # Take the first value of prediction
    output = prediction[0]
    print("trying to return prediction")
    print(prediction)
    #resp = {
    #    "sick": prediction[0]
    #}
    resp = prediction.tolist()
    return jsonify(resp)

if __name__ == '__main__':
    app.run(port=5000, debug=True)