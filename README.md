# WorldCupPredictor
## Application Function
A machine learning model designed to predict the world cup winner.  Complete with a UI that allows a user to change the teams and team statistics (ATT, DEF, MID, OVR) to produce a custom prediction.
## Implementation Methodology
We decided to use a simple linear regression for our model and gathered our training data by webscraping several different sites and compiling that data together into a single dataframe.  We then converted our model into a pkl file to make it easier to reference from the backend file and for our UI we used relatively simple html and connected everything together using a python file which converted the user-imputed data into a dataframe and passed that into our model.