from flask import Flask, request, jsonify, render_template,url_for
import sqlite3
import json
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


app = Flask(__name__)

# begin writing cursors for the two data bases "mnist" and "mnist_label"

# -------------  TEST DATA "mnist"

# db read for mnist data and corresponding cursor
mnist_db= sqlite3.connect('mnist.db')
cursor1 = mnist_db.cursor()

#creating list and then dataframe from mnist test data
d1 = cursor1.execute("SELECT * FROM mnist_test").fetchall()
mnist_datax = pd.DataFrame(d1)

# -------------  TEST LABEL "mnist_label"

mnist_label_db= sqlite3.connect('mnist_label.db')
cursor2 = mnist_label_db.cursor()

d2 = cursor2.execute("SELECT * FROM mnist_test_label").fetchall()
mnist_labely = pd.DataFrame(d2)

# -----------
# Import MLPclassifier by using pickle

# loading saved pickle model
filename = 'number_classifier.sav'
imported_num_classifier = pickle.load(open(filename, 'rb'))

global num_predict_df
num_predict_label = imported_num_classifier.predict(mnist_datax)
num_predict_df = pd.DataFrame(num_predict_label)
num_predict_json = num_predict_df.to_json()

@app.route('/')
def home():
    # simple string for home page
    return "MNIST Digits PredictionTool"

@app.route('/predict')
def prediction():
    # use global variable to pull json form of dataframe
    global num_predict_json

    return jsonify(num_predict_json)

@app.route('/accuracy')
def analysis():
    # use global variable again to pull variables
    global mnist_labely, num_predict_df

    # changed the decimal value to a percentage and then convert to string
    accuracy = accuracy_score(mnist_labely, num_predict_df)
    percentage = str(accuracy * 100)
  
    return("The accuracy of the prediction model on the test data is: " + percentage + "%")
    
app.run(debug=True)