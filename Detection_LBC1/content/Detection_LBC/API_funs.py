class N31_Noprep:
  #Load .mat files
  def mat_fun(dh,db):
    h = pd.DataFrame(dh['receivedsignalH_1'])
    h1 = h.astype('int64')
    h1['class'] = 0

    b = pd.DataFrame(db['receivedsignalBCH_2'])
    b1 = b.astype('int64')
    b1['class'] = 1
    dat=pd.concat([h1,b1],axis=0)
    return dat
  #Create Data slices
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
  #Model  
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
 

 class N15_NoPrep:
  def mat_f(dh,db):
    h = pd.DataFrame(dh['receivedsignalH_1'])
    h1 = h.astype('int64')
    h1['class'] = 0

    b = pd.DataFrame(db['receivedsignalBCH_2'])
    b1 = b.astype('int64')
    b1['class'] = 1
    dat=pd.concat([h1,b1],axis=0)
    return dat
  def dat_samp(dat1):
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
  def data_pre(dat):
    df = shuffle(dat)
    x=df.drop(['class'],axis=1)
    y=df['class']
    return x,y
## Model
  def model1():
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu',input_shape=(15,1)))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(1,activation='sigmoid'))
    return model

  def train(dat,e,b,s,d):
    mod=model1()
    checkpointer = ModelCheckpoint(filepath="model/"+str(s)+"_model_best_"+ str(d) +"P001.h5", monitor='accuracy',mode='max',verbose=1, save_best_only=True)
    opt1=Adam(learning_rate=0.001)
    mod.compile(loss='binary_crossentropy', optimizer=opt1, metrics=['accuracy'])
    x,y = data_pre(dat)
    mod.fit(x, y,batch_size=b, epochs=e,callbacks=[checkpointer],verbose=1)
    mod.save("model/"+str(s)+"_model_"+ str(d) +"P001.h5")
    return mod
  def test_pre(dh,db):
    dat=mat_f(dh,db)
    x=dat.drop(['class'],axis=1)
    y=dat['class']
    return x,y    
  def pred(model,x_t,y_t):
    pred_1 = (model.predict(x_t) > 0.5).astype(int)
    acc = accuracy_score(y_t, pred_1)
    return acc

    class third_four:
  
  ## Mapping the data
  def mapp(data):
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
  def mat_f(dh,db):
    h1=mapp(dh)
    h1['class'] = 0
    b1=mapp(db)
    b1['class'] = 1
    dat=pd.concat([h1,b1],axis=0)
    return dat
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
    d11=pd.concat([c1[181:1181],c2[181:1181]])
    return d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11
## Model
  def model1():
    model = Sequential()
    model.add(Dense(5, activation='relu',input_dim=5))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

  ## Mapping the data
  def mapp(data):
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

## Labelling
  def one_h(dat):
    x=dat.drop(['class'],axis=1)
    y=dat['class']
    #y = keras.utils.to_categorical(y, 2)
    return x,y
## Labelling
  def one_h_t(dat):
    x=dat.drop(['class'],axis=1)
    y=dat['class']
    #y = keras.utils.to_categorical(y, 2)
    return x,y


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

  def test_pre(dat):
    #dat=mat_f(dh,db)
    x=dat.drop(['class'],axis=1)
    y=dat['class']
    #y = keras.utils.to_categorical(y, 2)
    return x,y
  def mat_f(dh,db):
    h1=mapp(dh)
    h1['class'] = 0

    b1=mapp(db)
    b1['class'] = 1
    dat=pd.concat([h1,b1],axis=0)
    return dat
  ## Mapping the data
  def mapp(data):
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