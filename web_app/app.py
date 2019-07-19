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

scheduler = BackgroundScheduler()
scheduler.start()


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def get_page():
    d = json.loads(requests.get('https://galvanize-case-study-on-fraud.herokuapp.com/data_point').text)
    data = pd.DataFrame(columns=list(d.keys()))
    data.loc[0] = list(d.values())
    return(data)


low = 0
med = 0
high = 0
scheduler.add_job(get_page, 'interval', seconds=20)

@app.route('/')
def home():
    global low
    global med
    global high
    data = get_page()
    clean_data = clean(data.copy())
    model = joblib.load('models/rf_model.sav')
    prob = model.predict_proba(clean_data)[0][1]
    if prob <= .3:
        low += 1
        risk = 'Low'
    elif prob <= .6:
        med += 1
        risk = 'Medium'
    else:
        high += 1
        risk = 'High'
    return render_template('home.html',risk=risk,data=data,high=high,medium=med,low=low)


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
    
