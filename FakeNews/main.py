from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import argparse

import os
from flask import Flask, flash, request, redirect, url_for, render_template

from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageOps



app = Flask(__name__)




@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    url = text
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    for script in soup(["script", "style"]):
      script.extract()    # rip it out

    # get text
    text = soup.get_text()
    final_text = " ".join(text.split())

    input = [final_text]

    loaded_vectorizer = joblib.load('vectorizer.txt')

    tfidf_test=loaded_vectorizer.transform(input)

    loaded_model = joblib.load('final_model.sav')

    y_pred=loaded_model.predict(tfidf_test)

    return y_pred

app.run(host='0.0.0.0', port=8080)