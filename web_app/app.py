# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import requests
import pandas as pd
from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from clean_stream import clean
import joblib
from bs4 import BeautifulSoup

#Turn on scheduler to live steam data in 
scheduler = BackgroundScheduler()
scheduler.start()

#Initiate flask for web app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def get_page():
    """
    Revieves data from the specified url for prediction
    
    Returns:
        data (PandasDataFrame): 1 row DataFrame from url
    """
    d = json.loads(requests.get('URL GOES HERE').text)
    data = pd.DataFrame(columns=list(d.keys()))
    data.loc[0] = list(d.values())
    return(data)

#Counter for total of risk types assessed
low = 0
med = 0
high = 0
scheduler.add_job(get_page, 'interval', seconds=15)


@app.route('/')
def home():
    """
    Call functions and model to assess the data recieved and make predidction,
    renders home.html and counts ammount of each risk recieved.
    """
    global low
    global med
    global high
    data = get_page()
    data_clean = clean(data.copy())
    model = joblib.load('models/rf_model.sav')
    prob = model.predict_proba(data_clean)[0][1]
    perc = round(prob*100,2)
    if prob <= .3:
        low += 1
        risk = 'Low'
    elif prob <= .6:
        med += 1
        risk = 'Medium'
    else:
        high += 1
        risk = 'High'
    name = data.name.values[0]
    venue = data.venue_name.values[0]
    desc = data.description.values[0]
    clean_desc = BeautifulSoup(desc,'html.parser').get_text()\
    .replace('\n',' ').replace('\xa0','')\
    .replace("\'re"," are").replace("\'s"," is")
    
    return render_template('home.html', perc=perc, risk=risk, name=name, 
                           venue=venue, clean_desc=clean_desc, high=high, 
                           medium=med,low=low)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
    
