import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

#Importing .matfiles to pandas DataFrame
def matlab_file_import(dh,db):
  h = pd.DataFrame(dh['receivedsignalH_1'])
  h1 = h.astype('int64')
  h1['class'] = 0

  b = pd.DataFrame(db['receivedsignalBCH_2'])
  b1 = b.astype('int64')
  b1['class'] = 1
  dat=pd.concat([h1,b1],axis=0)
  return dat

#Data Concatenation- we have generated the dataset(.matfile) in a stepwise so we need arrange the dataset in stepwise before training NN model
def data_concatenation(dat1):
  c1=dat1[dat1['class'] == 0]
  c2=dat1[dat1['class'] == 1]
  d1=pd.concat([c1[:100],c2[:100]])
  d2=pd.concat([c1[100:300],c2[100:300]])
  d3=pd.concat([c1[300:600],c2[300:600]])
  d4=pd.concat([c1[600:1000],c2[600:1000]])
  d5=pd.concat([c1[1000:1500],c2[1000:1500]])
  d6=pd.concat([c1[1500:2500],c2[1500:2500]])
  d7=pd.concat([c1[2500:4500],c2[2500:4500]])
  d8=pd.concat([c1[4500:7500],c2[4500:7500]])
  d9=pd.concat([c1[7500:11500],c2[7500:11500]])
  d10=pd.concat([c1[11500:16500],c2[11500:16500]])
  return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10  

#Data shuffle-shuffle the dataset before training NN    
def data_shuffle(dat):
  df = shuffle(dat)
  x=df.drop(['class'],axis=1)
  y=df['class']
  return x,y  

## Proposed NN Model
def model1():
  model = Sequential()
  model.add(Conv1D(filters=8, kernel_size=3, activation='relu',input_shape=(15,1)))
  model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.20))
  model.add(Dense(1,activation='sigmoid'))
  return model  

## Training - here we will take model along with input data, no.of epoches,batch sie,set no.,dataset no. and save the model using checkpointer for the best model
def train(dat,Epochs,Batch_size,s,d):
  mod=model1()
  checkpointer = ModelCheckpoint(filepath="model/"+str(s)+"_model_best_"+ str(d) +"P001.h5", monitor='accuracy',mode='max',verbose=1, save_best_only=True)
  opt1=Adam(learning_rate=0.001)
  mod.compile(loss='binary_crossentropy', optimizer=opt1, metrics=['accuracy'])
  x,y = data_shuffle(dat)
  mod.fit(x, y,batch_size=Batch_size, epochs=Epochs,callbacks=[checkpointer],verbose=1)
  mod.save("model/"+str(s)+"_model_"+ str(d) +"P001.h5")
  return mod

#Test-Importing test data for prediction
def import_test_data(dh,db):
  dat=matlab_file_import(dh,db)
  x=dat.drop(['class'],axis=1)
  y=dat['class']
  return x,y 

##Prediction of Test Data
def Prediction(model,x_t,y_t):
  pred_1 = (model.predict(x_t) > 0.5).astype(int)
  acc = accuracy_score(y_t, pred_1)
  return acc  
