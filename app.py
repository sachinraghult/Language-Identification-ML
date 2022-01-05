#import libraries
import numpy as np
import string
import re
import pickle
from flask import Flask, render_template,request

#Initialize the flask App
app = Flask(__name__)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

def lang_detect(model, text):
    translate_table = dict((ord(char), None) for char in string.punctuation)

    global LangDetectModel

    if(model == 'lr'):
        lrLangDetectFile = open('models/lrmodel.pckl', 'rb')
        LangDetectModel = pickle.load(lrLangDetectFile)
        lrLangDetectFile.close()

    else:
        mnbLangDetectFile = open('models/mnbmodel.pckl', 'rb')
        LangDetectModel = pickle.load(mnbLangDetectFile)
        mnbLangDetectFile.close()

    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(translate_table)
    pred = LangDetectModel.predict([text])
    prob = LangDetectModel.predict_proba([text])
    return pred[0]


@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        data = [x for x in request.form.values()]
        result = lang_detect(data[0], data[1])
    return render_template('result.html', data0=data[0], data1=data[1], result=result)

if __name__ == "__main__":
    app.run(debug=True)
