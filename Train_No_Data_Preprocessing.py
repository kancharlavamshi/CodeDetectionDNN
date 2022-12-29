#!pip install mat73
## Import Data Files
import mat73
import pandas as pd
from N15_nopre import matlab_file_import,data_concatenation,train,import_test_data,Prediction 

#Importing .matfies
Hamming_train_data = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_1.mat')
Bch_train_data = mat73.loadmat('/content/BCH_RE_n_15_100_10k_1.mat')

Data = matlab_file_import(Hamming_train_data,Bch_train_data)

d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 = data_concatenation(Data)



Epochs=10  ## no.of epochs
Batch_size=3   ## batch_size

# Training 
model_1=train(d1,Epochs,Batch_size,'set1','1')
model_2=train(d2,Epochs,Batch_size,'set1','2')
model_3=train(d3,Epochs,Batch_size,'set1','3')
model_4=train(d4,Epochs,Batch_size,'set1','4')
model_5=train(d5,Epochs,Batch_size,'set1','5')
model_6=train(d6,Epochs,Batch_size,'set1','6')
model_7=train(d7,Epochs,Batch_size,'set1','7')
model_8=train(d8,Epochs,Batch_size,'set1','8')
model_9=train(d9,Epochs,Batch_size,'set1','9')
model_10=train(d10,Epochs,Batch_size,'set1','10')


##Testing

Hamming_test_data = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_test-1.mat')
Bch_test_data = mat73.loadmat('/content/BCH_RE_n_15_100_10k_test-1.mat')

codeword,label=import_test_data(Hamming_test_data,Bch_test_data)

accuracy_model_1=pred(model_1,codeword,label)
accuracy_model_2=pred(model_,codeword,label)
accuracy_model_3=pred(model_3,codeword,label)
accuracy_model_4=pred(model_4,codeword,label)
accuracy_model_5=pred(model_5,codeword,label)
accuracy_model_6=pred(model_6,codeword,label)
accuracy_model_7=pred(model_7,codeword,label)
accuracy_model_8=pred(model_8,codeword,label)
accuracy_model_9=pred(model_9,codeword,label)
accuracy_model_10=pred(model_10,codeword,label)

Accuracy=pd.concat([ pd.DataFrame([accuracy_model_1]),pd.DataFrame([accuracy_model_2]),  pd.DataFrame([accuracy_model_3]),pd.DataFrame([accuracy_model_4]), pd.DataFrame([accuracy_model_5]),pd.DataFrame([accuracy_model_6]),pd.DataFrame([accuracy_model_7]),
                      pd.DataFrame([accuracy_model_8]), pd.DataFrame([accuracy_model_9]), pd.DataFrame([accuracy_model_10])])
print(Accuracy)
