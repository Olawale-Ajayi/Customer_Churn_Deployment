import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel("C:/Users/admin/Desktop/Churn App/Telcom Data.xlsx")
df.drop_duplicates(inplace = True)
df = df[df["TotalCharges"]!=" "]
df["TotalCharges"] = df["TotalCharges"].astype(float)
Features = df.drop(["customerID", "Churn"], axis  = 1)
Target = df["Churn"]
#print(Features.columns)


#scaler = StandardScaler()
encoder = LabelEncoder()
Features['gender'] = encoder.fit_transform(Features["gender"])  # male: 1, female: 0
Features["PaymentMethod"]  = encoder.fit_transform(Features["PaymentMethod"])
mapping = {"No": 0, "Yes": 1}
Y = Target.map(mapping)
X = Features
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state =  42, shuffle = True)
Model = LogisticRegression(n_jobs = -1)
Model.fit(X_Train, Y_Train)
pred = Model.predict(X_Test)
score = accuracy_score(pred, Y_Test)
print(score)




pickle.dump(Model, open("Churn.pkl", "wb"))




