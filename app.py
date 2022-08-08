
# from unittest import skipUnless
from flask import Flask, render_template, request
# from flask import send_from_directory


# import numpy as np
# import pandas as pd
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import re
# import nltk
# from nltk.corpus import stopwords
# import string
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# import PyPDF2
# import pickle
from Model import ResumeModel

app = Flask(__name__)
model = ResumeModel()

@app.route('/', methods = ['GET'])
def Hello_world():
    return render_template('index.html')

@app.route('/', methods= ['POST'])
def predict():
    resumefile = request.files['resumefile']
    resume_path = "./resumes/" + resumefile.filename
    resumefile.save(resume_path)

    output = model.prediction(resume_path)

    # classification = f'Role of :{output}'


    return render_template('index.html', prediction = output)



if __name__ == '__main__':
    app.run(port=3000, debug=True)