"""
Created on Wed Jun 23 01:20:43 2021

@author: benit
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
######################################
# warnings.filterwarnings("ignore")


output_dir = 'C:/Users/52999/Desktop/MatrixOptim/OnlyBMV_Minutes/'

df = pd.read_csv("OnlyBMV_Minutes/KOF.csv")[["Datetime"]]
for x in range(df.shape[0]):
    df['Datetime'][x] = datetime.strptime(df.copy()['Datetime'][x][:19], '%Y-%m-%d %H:%M:%S') 

#k=0
# Read ALL CSV in folder
for filename in os.listdir(output_dir):
    if filename.endswith(".csv"):
        f = pd.read_csv('OnlyBMV_Minutes/' + filename, header=0)[["Datetime","Close"]]
        f.columns = ["Datetime", filename.strip(".csv")]
        for x in range(f.shape[0]):
            f['Datetime'][x] = datetime.strptime(f.copy()['Datetime'][x][:19], '%Y-%m-%d %H:%M:%S') 
        
        if (f.shape[0]>3000):
            df = pd.merge(df, f, "left")        
            
          
for y in df.columns:
    if ((df.isnull().sum()/df.shape[0])[y] >= 0.5):
        df = df.drop(y, axis=1)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(df.iloc[:,1:])
Imp_df = pd.DataFrame(imp.transform(df.iloc[:,1:]), columns = df.columns[1:])
Imp_df["date"] = df['Datetime']

RAWDataMatrix=Imp_df.dropna()
RAWDataMatrix.to_csv('Data_Storage/BMV/BMV.csv')


######################################
#Get returns
folder="Data_Storage/BMV/"
s1s=["BMV"]
for i in range(len(s1s)):
    s1=s1s[i]
    s1end=folder+s1+".csv"
    s1norm=folder+s1+"norm.csv"

    X =  pd.read_csv(s1end) # Matriz aleatoria Estructurada
    datenames = X["date"]
    X = X.set_index('date')
    X = X.drop(X.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 
    X = X.replace([np.inf, -np.inf, np.nan],0)
    Late=X.iloc[:-1,:].copy()
    Early=X.iloc[1:,:].copy()
    
    Late=Late.reset_index()
    Early=Early.reset_index()
    Late=Late.drop(Late.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 
    Early=Early.drop(Early.columns[[0]], axis=1)  # df.columns is zero-based pd.Index 
    Profits_2=(Early-Late)/Late
    X2=pd.DataFrame(Profits_2)
    X2=X2.set_index(datenames[1:])
    X2.to_csv(s1end)


    """    
    A1=np.corrcoef(X2.T)
    h=A1.shape[0]
    marker1=np.zeros(h)
    for i in range(h):
        for j in range(h):
            if(marker1[i]==0):
                if(abs(A1[i,j])>0.65 and i!=j):
                    marker1[i]=1
    independent0=list()
    for k in range(marker1.shape[0]):
        if marker1[k]==0:
            independent0.append(X2.columns[k])
    Profits_2=Profits_2[independent0]
    """
    X1=pd.DataFrame(Profits_2)


    object = StandardScaler()
    X1=object.fit_transform(Profits_2)
    X1=pd.DataFrame(X1)
    X1.columns=Profits_2.columns
    X1=X1.set_index(datenames[1:])
    X1.to_csv(s1norm)
    print(X1.shape)
    