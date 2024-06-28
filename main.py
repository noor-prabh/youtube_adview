import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

data_train = pd.read_csv("train.csv")
data_train.head()
data_train.shape
#(14999,9)

# Assigning each category a number for category feature
category={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_train["category"]=data_train["category"].map(category)
data_train.head()

#Removing character "F" present in data
data_train = data_train[data_train.views!="F"]
data_train = data_train[data_train.likes!="F"]
data_train = data_train[data_train.dislikes!="F"]
data_train = data_train[data_train.comment!="F"]

#Convert values to integers for views, likes, comments, dislikes and adview
data_train["views"]=pd.to_numeric(data_train["views"])
data_train["comment"]=pd.to_numeric(data_train["comment"])
data_train["likes"]=pd.to_numeric(data_train["likes"])
data_train["dislikes"]=pd.to_numeric(data_train["dislikes"])
data_train["adview"]=pd.to_numeric(data_train["adview"])

column_vidid = data_train['vidid']

#Encoding features like category, duration, vidid
from sklearn.preprocessing import LabelEncoder
data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])
data_train.head()

#convert time_in_sec for duration
import datetime
import time

def checki(x):
    y=x[2:]
    h=''
    m=''
    s=''
    mm=''
    P=['H','M','S']
    for i in y:
        if i not in P:
            mm+=i
        else:
            if(i=="H"):
                h=mm
                mm=''
            elif(i=="M"):
                m=mm
                mm=''
            else:
                s=mm
                mm=''
    if(h==''):
        h='00'
    if(m==''):
        m='00'
    if(s==''):
        s='00'
    bp = h+ ':'+m+':'+s
    return bp
tain = pd.read_csv("train.csv")
mp = pd.read_csv( "train.csv")["duration"]
time = mp.apply(checki)

def func_sec(time_string):
    h, m, s = time_string.split(':')
    return int(h)*3600 + int(m)*60 + int(s) 

time1 = time.apply(func_sec)
data_train["duration"] = time1
data_train.head()


#Visualization
    #individual plots
plt.hist(data_train["category"])
plt.show()
plt.plot(data_train["adview"])
plt.show()

#removing adview outliers i.e adview greater than 2000000
data_train = data_train[data_train["adview"]<2000000]


#heatmap
import seaborn as sns
f, ax = plt.subplots(figsize=(10,8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool)
            , cmap=sns.diverging_palette(220,10, as_cmap=True),square=True,ax=ax,annot=True)
plt.show()
              
#Splitting dataset
