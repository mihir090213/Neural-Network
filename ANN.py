#Importing Libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('D:\Dataset/Churn_Modelling.csv')
x= dataset.iloc[:,3:13].values
x
y= dataset.iloc[:,13].values 
y
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder  
label_encoder_x_1= LabelEncoder()  
x[:, 1]= label_encoder_x_1.fit_transform(x[:, 1]) 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x_2= LabelEncoder()  
x[:, 2]= label_encoder_x_2.fit_transform(x[:, 2])  
#Spliting data into test set and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#Data Scaling
from sklearn.preprocessing import StandardScaler 
sc= StandardScaler()  
x_train= sc.fit_transform(x_train)  
x_test= sc.transform(x_test) 
#Importing keras libraries
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
#Initialising the ANN
classifier=Sequential()
#Adding the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=10))
#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the ANN to the training set
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
#Predicting the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)
#Predicting the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)