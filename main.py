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
y_train = pd.DataFrame(data=data_train.iloc[:,1].values, columns=['target'])
data_train = data_train.drop(["adview"], axis=1)
data_train = data_train.drop(["vidid"], axis=1)
data_train.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_train, y_train, test_size=0.2, random_state=2)
x_train.shape

#normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train.mean()

#evaluation metrics
from sklearn import metrics
def print_error(x_test, x_train, model_name):
    prediction = model_name.predict(x_test)
    print('Mean absolute error :', metrics.mean_absolute_error(y_test, prediction))
    print('Mean squared error :', metrics.mean_squared_error(y_test, prediction))
    print('Root mean squared error :', np.sqrt(metrics.mean_squared_error(y_test, prediction))
          )

#linear regression
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
print_error(x_test, y_test, linear_regression)
#Mean absolute error : 3101.313562824354
#Mean squared error : 258641479.6389894
#Root mean squared error : 16082.334396442246

#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(x_train, y_train)
print_error(x_test, y_test, decision_tree)
#Mean absolute error : 2398.600068306011
#Mean squared error : 494257637.80635244
#root mean squared error : 22231.90585186867

#random forest regressor
from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
random_forest.fit(x_train, y_train)
print_error(x_test, y_test, random_forest)
#Mean absolute error : 2586.257158331853
#Mean squared error : 259690141.1950881
#Root mean squared error : 16114.904318521041

#support vector regressor
from sklearn.svm import SVR
supportvector_regressor = SVR()
y_train = y_train.values.ravel()
supportvector_regressor.fit(x_train, y_train)
print_error(x_test, y_test, supportvector_regressor )
#Mean absolute error : 1435.6157776650953
#Mean squared error : 260643200.93903348
#Root mean squared error : 16144.44799115267

#Artificial neural network
import keras
from keras.layers import Dense
ann = keras.models.Sequential([
                               Dense(6, activation="relu",
                               input_shape=x_train.shape[1:]),
                               Dense(6,activation="relu"),
                               Dense(1)           
                               ])

optimizer = keras.optimizers.Adam()
loss = keras.losses.mean_squared_error
ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])
history = ann.fit(x_train, y_train, epochs =100)
ann.summary()
print_error(x_test, y_test, ann)

#saving scikitlearn models
import joblib
joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")

#saving keras model
ann.save('ann_youtubeadview.keras')








