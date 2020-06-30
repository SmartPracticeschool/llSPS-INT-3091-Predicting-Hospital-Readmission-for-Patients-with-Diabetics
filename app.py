import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('project.pkl', 'rb'))

@app.route('/')
def home():
    #y_predict()
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    print('I am in y_predict function')
    x_test = [[int(x) for x in request.form.values()]]
    print(x_test)
    pipe=pickle.load('convert.save')  
    prediction = model.predict(pipe.transform(x_test))
    print(prediction)   
    output=prediction[0][0]
    if(output==0):
        pred=" is readmitted"
    else:
        pred=" is not readmitted"
    return render_template('index.html', prediction_text='patient {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
