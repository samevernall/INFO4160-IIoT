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

import os.path

# The if statement below checks if the databases exists
# if the car sales database does not exist, it will creat one for storage of future values
if os.path.exists('namecar_db.sqlite') == False:
    conn = sqlite3.connect('namecar_db.sqlite')
    cur = conn.cursor()
    cur.execute('CREATE TABLE car_sales (name VARCHAR, car_quality VARCHAR, buy_price VARCHAR, maint_price VARCHAR, doors VARCHAR, passengers VARCHAR, boot VARCHAR, safety VARCHAR)')
    conn.commit()
    conn.close()


app = Flask(__name__)

# Import Random Forest Classifier
filename = 'car_eval.sav'
car_eval = pickle.load(open(filename, 'rb'))

# Import blank feature column csv to instantiate car feature inputs 
# (1 by 21 df) -- 21 car features
# Begin index at column 0
carfeature_df = pd.read_csv('carfeature_input.csv', index_col=[0])

@app.route('/')
def home():
    return render_template("carcondition.html")

@app.route('/process',methods=['GET','POST'])
def read():
    global car_eval
    global carfeature_df

    # Pull in website inputs from user
    name = request.form.get('name')

    buy_price = request.form.get('buy_price')
    maint_price = request.form.get('maint_price')
    doors = request.form.get('doors')
    passengers = request.form.get('passengers')
    boot = request.form.get('boot')
    safety = request.form.get('safety')

    column_data = [buy_price, maint_price, doors, passengers, boot, safety]
    column_names = ['buy price', 'maint price', 'doors', 'passengers', 'boot', 'safety'] 

    # Making a dataframe with imported data from web
    raw_input = pd.DataFrame(data = [column_data], columns = column_names)

    # Creating get dummies column titles and making list from titles
    proper_labels = pd.get_dummies(data = raw_input, columns = column_names)
    features_list = list(proper_labels.columns.values)

    # Creating list of true column titles and zero data *
    # ** This is for resetting data values to zero when we run the predictive model again
    carfeature_list = list(carfeature_df.columns.values)
    zeros = np.zeros((1,21))

    # Create dataframe for feature input
    final_input = pd.DataFrame(data = zeros, columns = carfeature_list)
    # Set values to 1 where features list match true list on datafram
    final_input.loc[0, features_list] = 1


    # Use the car_eval Random Forest Classifier with dataframe input
    car_score = car_eval.predict(final_input)
    
    acceptability = np.array2string(car_score)
    
    # Below statements clean up string and make it a readable word
    if acceptability == "['unacc']":
        acceptability = 'unacceptable'
    if acceptability == "['acc']":
        acceptability = 'acceptable'
    if acceptability == "['good']":
        acceptability = 'good'
    if acceptability == "['vgood']":
        acceptability = 'very good'

    data = name + " your car is in [ " + acceptability + " ] condition"

    # recreate cursor
    conn = sqlite3.connect('namecar_db.sqlite')
    cur = conn.cursor()

    # save all input and output variable by commiting to databse

    cur.execute('INSERT INTO car_sales (name, car_quality, buy_price, maint_price, doors, passengers, boot, safety) VALUES (?, ?, ?, ? ,? ,? ,?, ?)',
            (name , acceptability, buy_price, maint_price, doors, passengers, boot, safety))
    conn.commit()
    conn.close()



    return render_template('carcondition.html',data=data)

app.run(host='127.0.0.1', port=5000)

