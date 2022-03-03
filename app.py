import pandas as pd
import numpy as np
import warnings
from flask import Flask, request, render_template
import pickle
# from Model import scaler

warnings.filterwarnings("ignore")


app = Flask(__name__)

model = pickle.load(open("Churn.pkl", "rb"))

@app.route("/")
def main():
    return render_template("index2.html")

@app.route("/predict", methods = ["POST"])
def home():
    gender = int(request.form["gender"])
    seniority = int(request.form["Seniority"])
    Payment = int(request.form["Payment"])
    MonthlyCharges = int(request.form["MCharges"])
    TotalCharges = int(request.form["TCharges"])

    # if type(MonthlyCharges) != str and type(TotalCharges) != str:

    #     MonthlyCharges = float(request.form["MCharges"])
    #     TotalCharges = float(request.form["TCharges"])
        
    X = np.array([[gender, seniority, Payment, MonthlyCharges, TotalCharges]])
    X = X.reshape(1,-1)
    #X[0,[3,4]] = scaler.transform(X[0,[3,4]].reshape(1,-1))
    #cat = ColumnEncoder()
    #features = cat.fit_transform(X)
    pred = model.predict(X)
    return render_template("index.html", data = pred)

    # else:
    #     error = "You entered a wrong Monthly and Total Charges. Please enter a number for both feeds to continue"
    #     err = str(type(MonthlyCharges)) + "  " +  str(type(TotalCharges))
    #     return render_template("fail.html", error = err)
         




if __name__ == "__main__":
    app.run(debug = True, use_reloader = False)