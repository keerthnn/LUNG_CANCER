# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:28:00 2020

@author: DELL
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)
model1=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    print(int_features)
    final_features=[np.array(int_features)]
    #feature_name=['age', 'menopause', 'tumorsize', 'invnodes', 'nodecaps', 'degmalig', 'breast', 'breastquad','irradiat']
    #df=pd.DataFrame(final_features, columns=feature_name)
    output=model1.predict(final_features)
    if output == 0:
        outp="negative"
    else:
        outp="positive"
    if outp=="positive":
        #return render_template('index.html',prediction_text='Your result is {}'.format(outp))
        return render_template('index.html',prediction_text='YES')
    else:
        return render_template('index.html',prediction_text='NO')


if __name__=="__main__":
    app.run(debug=True)