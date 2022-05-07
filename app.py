#import required packages
from flask import Flask, render_template, request
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

# create a function Home that will return index.html(which contains html form)
# index.html file is created seperately
def Home():
    return render_template('index.html')

# HTTP POST request method
# define the route for post method 
@app.route("/predict", methods=['POST'])

# define the predict function which is going to predict the results from ml model based on the given values through html form
def predict():

    # initialize dataframe
    game = pd.DataFrame(columns=['Team', 'ATT', 'MID', 'DEF', 'OVR'])
    
    if request.method == 'POST':
        # Use request.form to get the data from html form through post method.
        Team1 = request.form['Team1']
        ATT1 = int(request.form['ATT1'])
        MID1 = int(request.form['MID1'])
        DEF1 = int(request.form['DEF1'])
        OVR1 = int(request.form['OVR1'])
        game = game.append({'Team': Team1, 'ATT': ATT1, 'MID': MID1, 'DEF': DEF1, 'OVR': OVR1}, ignore_index=True)
        Team2 = request.form['Team2']
        ATT2 = int(request.form['ATT2'])
        MID2 = int(request.form['MID2'])
        DEF2 = int(request.form['DEF2'])
        OVR2 = int(request.form['OVR2'])
        game = game.append({'Team': Team2, 'ATT': ATT2, 'MID': MID2, 'DEF': DEF2, 'OVR': OVR2}, ignore_index=True)
        Team3 = request.form['Team3']
        ATT3 = int(request.form['ATT3'])
        MID3 = int(request.form['MID3'])
        DEF3 = int(request.form['DEF3'])
        OVR3 = int(request.form['OVR3'])
        game = game.append({'Team': Team3, 'ATT': ATT3, 'MID': MID3, 'DEF': DEF3, 'OVR': OVR3}, ignore_index=True)
        Team4 = request.form['Team4']
        ATT4 = int(request.form['ATT4'])
        MID4 = int(request.form['MID4'])
        DEF4 = int(request.form['DEF4'])
        OVR4 = int(request.form['OVR4'])
        game = game.append({'Team': Team4, 'ATT': ATT4, 'MID': MID4, 'DEF': DEF4, 'OVR': OVR4}, ignore_index=True)
        Team5 = request.form['Team5']
        ATT5 = int(request.form['ATT5'])
        MID5 = int(request.form['MID5'])
        DEF5 = int(request.form['DEF5'])
        OVR5 = int(request.form['OVR5'])
        game = game.append({'Team': Team5, 'ATT': ATT5, 'MID': MID5, 'DEF': DEF5, 'OVR': OVR5}, ignore_index=True)
        Team6 = request.form['Team6']
        ATT6 = int(request.form['ATT6'])
        MID6 = int(request.form['MID6'])
        DEF6 = int(request.form['DEF6'])
        OVR6 = int(request.form['OVR6'])
        game = game.append({'Team': Team6, 'ATT': ATT6, 'MID': MID6, 'DEF': DEF6, 'OVR': OVR6}, ignore_index=True)
        Team7 = request.form['Team7']
        ATT7 = int(request.form['ATT7'])
        MID7 = int(request.form['MID7'])
        DEF7 = int(request.form['DEF7'])
        OVR7 = int(request.form['OVR7'])
        game = game.append({'Team': Team7, 'ATT': ATT7, 'MID': MID7, 'DEF': DEF7, 'OVR': OVR7}, ignore_index=True)
        Team8 = request.form['Team8']
        ATT8 = int(request.form['ATT8'])
        MID8 = int(request.form['MID8'])
        DEF8 = int(request.form['DEF8'])
        OVR8 = int(request.form['OVR8'])
        game = game.append({'Team': Team8, 'ATT': ATT8, 'MID': MID8, 'DEF': DEF8, 'OVR': OVR8}, ignore_index=True)
        Team9 = request.form['Team9']
        ATT9 = int(request.form['ATT9'])
        MID9 = int(request.form['MID9'])
        DEF9 = int(request.form['DEF9'])
        OVR9 = int(request.form['OVR9'])
        game = game.append({'Team': Team9, 'ATT': ATT9, 'MID': MID9, 'DEF': DEF9, 'OVR': OVR9}, ignore_index=True)
        Team10 = request.form['Team10']
        ATT10 = int(request.form['ATT10'])
        MID10 = int(request.form['MID10'])
        DEF10 = int(request.form['DEF10'])
        OVR10 = int(request.form['OVR10'])
        game = game.append({'Team': Team10, 'ATT': ATT10, 'MID': MID10, 'DEF': DEF10, 'OVR': OVR10}, ignore_index=True)
        Team11 = request.form['Team11']
        ATT11 = int(request.form['ATT11'])
        MID11 = int(request.form['MID11'])
        DEF11 = int(request.form['DEF11'])
        OVR11 = int(request.form['OVR11'])
        game = game.append({'Team': Team11, 'ATT': ATT11, 'MID': MID11, 'DEF': DEF11, 'OVR': OVR11}, ignore_index=True)
        Team12 = request.form['Team12']
        ATT12 = int(request.form['ATT12'])
        MID12 = int(request.form['MID12'])
        DEF12 = int(request.form['DEF12'])
        OVR12 = int(request.form['OVR12'])
        game = game.append({'Team': Team12, 'ATT': ATT12, 'MID': MID12, 'DEF': DEF12, 'OVR': OVR12}, ignore_index=True)
        Team13 = request.form['Team13']
        ATT13 = int(request.form['ATT13'])
        MID13 = int(request.form['MID13'])
        DEF13 = int(request.form['DEF13'])
        OVR13 = int(request.form['OVR13'])
        game = game.append({'Team': Team13, 'ATT': ATT13, 'MID': MID13, 'DEF': DEF13, 'OVR': OVR13}, ignore_index=True)
        Team14 = request.form['Team14']
        ATT14 = int(request.form['ATT14'])
        MID14 = int(request.form['MID14'])
        DEF14 = int(request.form['DEF14'])
        OVR14 = int(request.form['OVR14'])
        game = game.append({'Team': Team14, 'ATT': ATT14, 'MID': MID14, 'DEF': DEF14, 'OVR': OVR14}, ignore_index=True)
        Team15 = request.form['Team15']
        ATT15 = int(request.form['ATT15'])
        MID15 = int(request.form['MID15'])
        DEF15 = int(request.form['DEF15'])
        OVR15 = int(request.form['OVR15'])
        game = game.append({'Team': Team15, 'ATT': ATT15, 'MID': MID15, 'DEF': DEF15, 'OVR': OVR15}, ignore_index=True)
        Team16 = request.form['Team16']
        ATT16 = int(request.form['ATT16'])
        MID16 = int(request.form['MID16'])
        DEF16 = int(request.form['DEF16'])
        OVR16 = int(request.form['OVR16'])
        game = game.append({'Team': Team16, 'ATT': ATT16, 'MID': MID16, 'DEF': DEF16, 'OVR': OVR16}, ignore_index=True)
        Team17 = request.form['Team17']
        ATT17 = int(request.form['ATT17'])
        MID17 = int(request.form['MID17'])
        DEF17 = int(request.form['DEF17'])
        OVR17 = int(request.form['OVR17'])
        game = game.append({'Team': Team17, 'ATT': ATT17, 'MID': MID17, 'DEF': DEF17, 'OVR': OVR17}, ignore_index=True)
        Team18 = request.form['Team18']
        ATT18 = int(request.form['ATT18'])
        MID18 = int(request.form['MID18'])
        DEF18 = int(request.form['DEF18'])
        OVR18 = int(request.form['OVR18'])
        game = game.append({'Team': Team18, 'ATT': ATT18, 'MID': MID18, 'DEF': DEF18, 'OVR': OVR18}, ignore_index=True)
        Team19 = request.form['Team19']
        ATT19 = int(request.form['ATT19'])
        MID19 = int(request.form['MID19'])
        DEF19 = int(request.form['DEF19'])
        OVR19 = int(request.form['OVR19'])
        game = game.append({'Team': Team19, 'ATT': ATT19, 'MID': MID19, 'DEF': DEF19, 'OVR': OVR19}, ignore_index=True)
        Team20 = request.form['Team20']
        ATT20 = int(request.form['ATT20'])
        MID20 = int(request.form['MID20'])
        DEF20 = int(request.form['DEF20'])
        OVR20 = int(request.form['OVR20'])
        game = game.append({'Team': Team20, 'ATT': ATT20, 'MID': MID20, 'DEF': DEF20, 'OVR': OVR20}, ignore_index=True)
        Team21 = request.form['Team21']
        ATT21 = int(request.form['ATT21'])
        MID21 = int(request.form['MID21'])
        DEF21 = int(request.form['DEF21'])
        OVR21 = int(request.form['OVR21'])
        game = game.append({'Team': Team21, 'ATT': ATT21, 'MID': MID21, 'DEF': DEF21, 'OVR': OVR21}, ignore_index=True)
        Team22 = request.form['Team22']
        ATT22 = int(request.form['ATT22'])
        MID22 = int(request.form['MID22'])
        DEF22 = int(request.form['DEF22'])
        OVR22 = int(request.form['OVR22'])
        game = game.append({'Team': Team22, 'ATT': ATT22, 'MID': MID22, 'DEF': DEF22, 'OVR': OVR22}, ignore_index=True)
        Team23 = request.form['Team23']
        ATT23 = int(request.form['ATT23'])
        MID23 = int(request.form['MID23'])
        DEF23 = int(request.form['DEF23'])
        OVR23 = int(request.form['OVR23'])
        game = game.append({'Team': Team23, 'ATT': ATT23, 'MID': MID23, 'DEF': DEF23, 'OVR': OVR23}, ignore_index=True)
        Team24 = request.form['Team24']
        ATT24 = int(request.form['ATT24'])
        MID24 = int(request.form['MID24'])
        DEF24 = int(request.form['DEF24'])
        OVR24 = int(request.form['OVR24'])
        game = game.append({'Team': Team24, 'ATT': ATT24, 'MID': MID24, 'DEF': DEF24, 'OVR': OVR24}, ignore_index=True)
        Team25 = request.form['Team25']
        ATT25 = int(request.form['ATT25'])
        MID25 = int(request.form['MID25'])
        DEF25 = int(request.form['DEF25'])
        OVR25 = int(request.form['OVR25'])
        game = game.append({'Team': Team25, 'ATT': ATT25, 'MID': MID25, 'DEF': DEF25, 'OVR': OVR25}, ignore_index=True)
        Team26 = request.form['Team26']
        ATT26 = int(request.form['ATT26'])
        MID26 = int(request.form['MID26'])
        DEF26 = int(request.form['DEF26'])
        OVR26 = int(request.form['OVR26'])
        game = game.append({'Team': Team26, 'ATT': ATT26, 'MID': MID26, 'DEF': DEF26, 'OVR': OVR26}, ignore_index=True)
        Team27 = request.form['Team27']
        ATT27 = int(request.form['ATT27'])
        MID27 = int(request.form['MID27'])
        DEF27 = int(request.form['DEF27'])
        OVR27 = int(request.form['OVR27'])
        game = game.append({'Team': Team27, 'ATT': ATT27, 'MID': MID27, 'DEF': DEF27, 'OVR': OVR27}, ignore_index=True)
        Team28 = request.form['Team28']
        ATT28 = int(request.form['ATT28'])
        MID28 = int(request.form['MID28'])
        DEF28 = int(request.form['DEF28'])
        OVR28 = int(request.form['OVR28'])
        game = game.append({'Team': Team28, 'ATT': ATT28, 'MID': MID28, 'DEF': DEF28, 'OVR': OVR28}, ignore_index=True)
        Team29 = request.form['Team29']
        ATT29 = int(request.form['ATT29'])
        MID29 = int(request.form['MID29'])
        DEF29 = int(request.form['DEF29'])
        OVR29 = int(request.form['OVR29'])
        game = game.append({'Team': Team29, 'ATT': ATT29, 'MID': MID29, 'DEF': DEF29, 'OVR': OVR29}, ignore_index=True)
        Team30 = request.form['Team30']
        ATT30 = int(request.form['ATT30'])
        MID30 = int(request.form['MID30'])
        DEF30 = int(request.form['DEF30'])
        OVR30 = int(request.form['OVR30'])
        game = game.append({'Team': Team30, 'ATT': ATT30, 'MID': MID30, 'DEF': DEF30, 'OVR': OVR30}, ignore_index=True)
        Team31 = request.form['Team31']
        ATT31 = int(request.form['ATT31'])
        MID31 = int(request.form['MID31'])
        DEF31 = int(request.form['DEF31'])
        OVR31 = int(request.form['OVR31'])
        game = game.append({'Team': Team31, 'ATT': ATT31, 'MID': MID31, 'DEF': DEF31, 'OVR': OVR31}, ignore_index=True)
        Team32 = request.form['Team32']
        ATT32 = int(request.form['ATT32'])
        MID32 = int(request.form['MID32'])
        DEF32 = int(request.form['DEF32'])
        OVR32 = int(request.form['OVR32'])
        game = game.append({'Team': Team32, 'ATT': ATT32, 'MID': MID32, 'DEF': DEF32, 'OVR': OVR32}, ignore_index=True)
        
        prediction=model.predict([[game]])
        
        # condition for invalid values
        if prediction == None:
            return render_template('index.html',prediction_text="Unable to compute result")
        
        # condition for prediction when values are valid
        else:
            return render_template('index.html',prediction_text="Using given data, {} will win the World Cup".format(prediction))
        
    # html form to be displayed on screen when no values are inserted; without any output or prediction
    else:
        return render_template('index.html')


if __name__=="__main__":
    # run method starts our web service
    # Debug : as soon as I save anything in my structure, server should start again
    app.run(debug=True)