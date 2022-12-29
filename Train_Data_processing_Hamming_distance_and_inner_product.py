import mat73
import pandas as pd
from third_fourth_prep import  matlab_file_import_3,data_concatenation,train,import_test_data_3,Prediction 

#Importing .matfies
Hammining_train_data  = mat73.loadmat('/content/Hamm_H-D_5_100_n(15)_P001_1.mat')
Bch_train_data  = mat73.loadmat('/content/BCH_H-D_5_100_n(15)_P001_1.mat')

Data =  matlab_file_import_3(Hamming_train_data,Bch_train_data )

d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 = data_concatenation(Data)



Epochs=10  ## epochs
Batch_size=1   ## batch_size

# Training 
model_1=train(d1,Epochs,Batch_size)
model_2=train(d2,Epochs,Batch_size)
model_3=train(d3,Epochs,Batch_size)
model_4=train(d4,Epochs,Batch_size)
model_5=train(d5,Epochs,Batch_size)
model_6=train(d6,Epochs,Batch_size)
model_7=train(d7,Epochs,Batch_size)
model_8=train(d8,Epochs,Batch_size)
model_9=train(d9,Epochs,Batch_size)
model_10=train(d10,Epochs,Batch_size)


##Testing

Hammining_test_data = mat73.loadmat('/content/Hamm_H-D_5_100_n(15)_P001_test_1.mat')
Bch_test_data = mat73.loadmat('/content/BCH_H-D_5_100_n(15)_P001_test_1.mat')

codeword,label=import_test_data_3(ht_1,bt_1)

acc1=Prediction(model_1,codeword,label)
acc2=Prediction(model_2,codeword,label)
acc3=Prediction(model_3,codeword,label)
acc4=Predictionmodel_4,codeword,label)
acc5=Prediction(model_5,codeword,label)
acc6=Prediction(model_6,codeword,label)
acc7=Prediction(model_7,codeword,label)
acc8=Prediction(model_8,codeword,label)
acc9=Prediction(model_9,codeword,label)
acc10=Prediction(model_10,codeword,label)

accuracy_1=pd.concat([ pd.DataFrame([acc1]),pd.DataFrame([acc2]),  pd.DataFrame([acc3]),pd.DataFrame([acc4]), pd.DataFrame([acc5]),pd.DataFrame([acc6]),pd.DataFrame([acc7]),
                      pd.DataFrame([acc8]), pd.DataFrame([acc9]), pd.DataFrame([acc10])])
print(accuracy_1)
