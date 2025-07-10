import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #for rendering result on html ui
    int_features=[int(x) for x in request.form.values()]
    final_feature=[np.array(int_features)]
    prediction = model.predict(final_feature)

    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='employee salary {} aed'.format(output))

if __name__=="__main__":
    app.run(debug=True)