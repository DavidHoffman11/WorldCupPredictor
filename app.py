#import required packages
from zlib import DEF_BUF_SIZE
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd

# create a Flask object
app = Flask("WorldCup_model")

# load the ml model which we have saved earlier in .pkl format
model = pickle.load(open('world_cup_model.pkl', 'rb'))

# define the route(basically url) to which we need to send http request
# HTTP GET request method
@app.route('/',methods=['GET'])

# HTTP POST request method
# define the route for post method 
@app.route("/predict", methods=['POST'])

# create a function Home that will return index.html(which contains html form)
# index.html file is created seperately
def Home():
    return render_template('index.html')

# define the predict function which is going to predict the results from ml model based on the given values through html form
def predict():

    # initialize dataframe
    game = pd.DataFrame(columns=['Team', 'ATT', 'MID', 'DEF', 'OVR'])
    i = 1

    if request.method == 'POST':
        # Use request.form to get the data from html form through post method.
        while (i < 33):
            Team = request.form['Team'+i]
            ATT = int(request.form['ATT'+i])
            MID = int(request.form['MID'+i])
            DEF = int(request.form['DEF'+i])
            OVR = int(request.form['OVR'+i])
            game = game.append({'Team': Team, 'ATT': ATT, 'MID': MID, 'DEF': DEF, 'OVR': OVR}, ignore_index=True)
        
        prediction=model.predict([[game]])
        
        #condition for invalid values
        if prediction<0:
            return render_template('index.html',prediction_text="Unable to compute result")
        
        #condition for prediction when values are valid
        else:
            return render_template('index.html',prediction_text="Using given data, {} will win the World Cup".format(prediction))
        
    #html form to be displayed on screen when no values are inserted; without any output or prediction
    else:
        return render_template('index.html')


if __name__=="__main__":
    # run method starts our web service
    # Debug : as soon as I save anything in my structure, server should start again
    app.run(debug=True)