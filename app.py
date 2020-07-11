import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from flask import Flask,render_template,request
import pickle

obj1=pickle.load(open('word1 (1).pkl','rb'))
words=obj1.get_feature_names()
clf3=pickle.load(open('model2 (1).pkl','rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    review=request.form.get('review')
    sentiment=tr_review(review)
    print("The sentiment is",sentiment)
    if sentiment==0:
        return render_template('index.html',sentiment=-1)
    if sentiment==1:
        return render_template('index.html', sentiment=1)
def tr_review(r):
    hell = []
    for i in words:
        hell.append(r.count(i))
    X1 = np.array(hell).reshape(1, 5000)
    X2=clf3.predict(X1)[0]
    print(X2)
    return X2
if __name__=="__main__":
    app.run(debug=True)