from django.shortcuts import render
import pandas as pd
import numpy as np
from joblib import load

model=load('./savedModels/model.joblib')
def home(request):
    # if request.method=='POST':
    x=pd.read_csv('./predictor/cleanx.csv')
    x=x.drop('abc',axis='columns')
    # print(x)
    if request.method=='POST':
        location=request.POST['location']
        sqft=request.POST['sqft']
        bath=request.POST['bath']
        BHK=request.POST['bhk']

        location_index=np.where(x.columns==location)[0][0]
        X=np.zeros(len(x.columns))
        X[0]= sqft
        X[1]= bath
        X[2]= BHK
        if location_index>=0:
            X[location_index]=1
        result= model.predict([X])[0]
        
        print(result)
        return render(request,'home.html',{'result':result, 'BHK':BHK, 'bath':bath, 'sqft':sqft, 'location':location})

    return render(request,'home.html')
