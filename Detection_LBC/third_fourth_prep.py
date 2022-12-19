import pandas as pd
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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

  ## Mapping the data
def mapp3(data):
  h=pd.DataFrame(data['dist_h'])
  data=np.array(h)
  df = list()
  for i in range(len(data)):
    if data[i] == 0:
      df.append([9,9,9,9,9])
    else:
      df.append([1,1,1,1,1])
  
  df = pd.DataFrame(df)
  return df
#Importing .matfiles to pandas DataFrame
def mat_f3(dh,db):
  h1=mapp3(dh)
  h1['class'] = 0
  b1=mapp3(db)
  b1['class'] = 1
  dat=pd.concat([h1,b1],axis=0)
  return dat  
#Importing .matfiles to pandas DataFrame
def mat_f4(dh,db):
  h1=mapp4(dh)
  h1['class'] = 0
  b1=mapp4(db)
  b1['class'] = 1
  dat=pd.concat([h1,b1],axis=0)
  return dat 


#Data Concatenation
def dat_samp(dat1):
  c1=dat1[dat1['class'] == 0]
  c2=dat1[dat1['class'] == 1]
  d1=pd.concat([c1[:10],c2[:10]])
  d2=pd.concat([c1[10:25],c2[10:25]])
  d3=pd.concat([c1[25:41],c2[25:41]])
  d4=pd.concat([c1[41:58],c2[41:58]])
  d5=pd.concat([c1[58:76],c2[58:76]])
  d6=pd.concat([c1[76:95],c2[76:95]])
  d7=pd.concat([c1[95:115],c2[95:115]])
  d8=pd.concat([c1[115:136],c2[115:136]])
  d9=pd.concat([c1[136:158],c2[136:158]])
  d10=pd.concat([c1[158:181],c2[158:181]])
  #d11=pd.concat([c1[181:1181],c2[181:1181]])
  return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,#d11

## Model
def model1():
  model = Sequential()
  model.add(Dense(5, activation='relu',input_dim=5))
  model.add(Dense(1, activation='relu'))
  model.add(Dense(1,activation='sigmoid'))
  return model
## Labelling
def one_h(dat):
  x=dat.drop(['class'],axis=1)
  y=dat['class']
  #y = keras.utils.to_categorical(y, 2)
  return x,y
#Prediction
def pred(model,x_t,y_t):
  pred_1 = (model.predict(x_t) > 0.5).astype(int)
  acc = accuracy_score(y_t, pred_1)
    #pred_1 = model.predict(x_t) 
    #y_cls_1 = pred_1.argmax(axis=-1)
    #acc = accuracy_score(y_t, y_cls_1)  
  return acc

## Training
def train(dat,e,b):
  mod=model1()
  opt1=Adam(learning_rate=0.001)
  mod.compile(loss='binary_crossentropy', optimizer=opt1, metrics=['accuracy'])
  x,y = one_h(dat)
  mod.fit(x, y,batch_size=b, epochs=e,verbose=0)
  return mod
#Test
def test_pre3(dh,db):
  dat=mat_f3(dh,db)
  x=dat.drop(['class'],axis=1)
  y=dat['class']
  #y = keras.utils.to_categorical(y, 2)
  return x,y  
#Test
def test_pre4(dh,db):
  dat=mat_f4(dh,db)
  x=dat.drop(['class'],axis=1)
  y=dat['class']
  #y = keras.utils.to_categorical(y, 2)
  return x,y  
#Importing .matfiles to pandas DataFrame
def mat_f(dh,db):
  h1=mapp(dh)
  h1['class'] = 0

  b1=mapp(db)
  b1['class'] = 1
  dat=pd.concat([h1,b1],axis=0)
  return dat  
  ## Mapping the data
def mapp4(data):
  h=pd.DataFrame(data['dist_4'])
  data=np.array(h)
  df = list()
  for i in range(len(data)):
    if data[i] == 0:
      df.append([9,9,9,9,9])
    else:
      df.append([1,1,1,1,1])
  df = pd.DataFrame(df)
  return df   
