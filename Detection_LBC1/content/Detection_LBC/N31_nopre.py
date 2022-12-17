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


def mat_fun(dh,db):
  h = pd.DataFrame(dh['receivedsignalH_1'])
  h1 = h.astype('int64')
  h1['class'] = 0

  b = pd.DataFrame(db['receivedsignalBCH_2'])
  b1 = b.astype('int64')
  b1['class'] = 1
  dat=pd.concat([h1,b1],axis=0)
  return dat
def dat_samp_1(dat1):
  c1=dat1[dat1['class'] == 0]
  c2=dat1[dat1['class'] == 1]
  d1=pd.concat([c1[:10000],c2[:10000]])
  d2=pd.concat([c1[10000:30000],c2[10000:30000]])
  d3=pd.concat([c1[30000:60000],c2[30000:60000]])
  d4=pd.concat([c1[60000:100000],c2[60000:100000]])
  d5=pd.concat([c1[100000:150000],c2[100000:150000]])
  d6=pd.concat([c1[150000:210000],c2[150000:210000]])
  d7=pd.concat([c1[210000:280000],c2[210000:280000]])
  d8=pd.concat([c1[280000:360000],c2[280000:360000]])
  d9=pd.concat([c1[360000:450000],c2[360000:450000]])
  d10=pd.concat([c1[450000:550000],c2[450000:550000]])
  return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10    
 #Shuffle Data 
def data_shuff(dat):
  df = shuffle(dat)
  x=df.drop(['class'],axis=1)
  y=df['class']
  return x,y   
def model1():
  model = Sequential()
  model.add(Conv1D(filters=8, kernel_size=3, activation='relu',input_shape=(31,1)))
  model.add(Conv1D(filters=12, kernel_size=7, activation='relu'))
  model.add(Flatten())
  model.add(Dense(72, activation='relu'))
  model.add(Dense(18, activation='relu'))
  model.add(Dropout(0.10))
  model.add(Dense(1,activation='sigmoid'))
  return model
#Train the model (e=epochs,b=batch_size,s=dataset_no.,d=dataset_batch_no.)
def train(dat,e,b,s,d):
  x_t,y_t=test_pre(ht_1,bt_1)
  mod=model1()
  opt1=Adam(learning_rate=0.0001)
  mod.compile(loss='binary_crossentropy', optimizer=opt1, metrics=['accuracy'])
  x,y = data_pre(dat)
  mod.fit(x, y,batch_size=b,shuffle=True, epochs=e,verbose=1, validation_data=(x_t,y_t))
  mod.save("model/"+str(s)+"_model_"+ str(d) +"P001.h5")
  return mod  

    #Create Test Data 
def test_pre(dh,db):
  dat=mat_f(dh,db)
  x=dat.drop(['class'],axis=1)
  y=dat['class']
  return x,y


#Prediction
def pred(model,x_t,y_t):
  pred_1 = (model.predict(x_t) > 0.5).astype(int)
  acc = accuracy_score(y_t, pred_1)
  return acc