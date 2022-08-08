import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
import nltk
from nltk.corpus import stopwords
# import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import pickle



class ResumeModel:
    def __init__(self):
        self.classifier_model = self.load_model()
        self.word_vec = self.load_word_vec()
        self.label_encoder = self.load_label_encoder()
    
    def load_model(self):
        model = OneVsRestClassifier(KNeighborsClassifier())
        with open('model.pkl', "rb") as f:
            model = pickle.load(f)

        return model
    
    def load_word_vec(self):
        word_vec = TfidfVectorizer(sublinear_tf=True, stop_words='english')
        with open('word_vectorizer.pkl', "rb") as f:
            word_vec = pickle.load(f)
        
        return word_vec
    
    def load_label_encoder(self):
        le = LabelEncoder()
        with open('label_en.pkl', "rb") as f:
            le = pickle.load(f)
        
        return le
    
    def cleanResume(self, resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText

    
    def prediction(self, pdf_path):
        pdffile = open(pdf_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdffile)
       
        text =str()
        num_pages = pdfReader.numPages
        for page in range(num_pages):
            pageObj = pdfReader.getPage(page)

            text += pageObj.extract_text()
        
        pdffile.close()

        clean_text = [self.cleanResume(text)]
        word_features = self.word_vec.transform(clean_text)
        prediction = self.classifier_model.predict(word_features)
        output = self.label_encoder.inverse_transform(prediction)[0]

        return output


