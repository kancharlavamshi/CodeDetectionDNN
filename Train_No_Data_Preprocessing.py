#!pip install mat73
## Import Data Files
import mat73
import pandas as pd
from N15_nopre import mat_f,dat_samp,train,test_pre,pred 

#Importing .matfies
Hamming_train_data = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_1.mat')
Bch_train_data = mat73.loadmat('/content/BCH_RE_n_15_100_10k_1.mat')

Data = mat_f(Hamming_train_data,Bch_train_data)

d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 = dat_samp(Data)
#uncomment if you want to check the dataset shape 
##d1.shape,d2.shape,d3.shape,d4.shape,d5.shape,d6.shape,d7.shape,d8.shape,d9.shape,d10.shape


epochs=10  ## no.of epochs
batch_size=3   ## batch_size

# Training 
model_1=train(d1,e,b,'set1','1')
model_2=train(d2,e,b,'set1','2')
model_3=train(d3,e,b,'set1','3')
model_4=train(d4,e,b,'set1','4')
model_5=train(d5,e,b,'set1','5')
model_6=train(d6,e,b,'set1','6')
model_7=train(d7,e,b,'set1','7')
model_8=train(d8,e,b,'set1','8')
model_9=train(d9,e,b,'set1','9')
model_10=train(d10,e,b,'set1','10')


##Testing

Hamming_test_data = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_test-1.mat')
Bch_test_data = mat73.loadmat('/content/BCH_RE_n_15_100_10k_test-1.mat')

codeword,label=test_pre(Hamming_test_data,Bch_test_data)

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
