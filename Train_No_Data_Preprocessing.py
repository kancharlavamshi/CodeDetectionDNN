#!pip install mat73
## Import Data Files
import mat73
import pandas as pd
from N15_nopre import mat_f,dat_samp,train,test_pre,pred 

#Importing .matfies
dh_1 = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_1.mat')
db_1 = mat73.loadmat('/content/BCH_RE_n_15_100_10k_1.mat')

dat1 = mat_f(dh_1,db_1)

d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 = dat_samp(dat1)
d1.shape,d2.shape,d3.shape,d4.shape,d5.shape,d6.shape,d7.shape,d8.shape,d9.shape,d10.shape


e=10  ## epochs
b=3   ## batch_size

# Training 
mod1=train(d1,e,b,'set1','1')
mod2=train(d2,e,b,'set1','2')
mod3=train(d3,e,b,'set1','3')
mod4=train(d4,e,b,'set1','4')
mod5=train(d5,e,b,'set1','5')
mod6=train(d6,e,b,'set1','6')
mod7=train(d7,e,b,'set1','7')
mod8=train(d8,e,b,'set1','8')
mod9=train(d9,e,b,'set1','9')
mod10=train(d10,e,b,'set1','10')


##Testing

ht_1 = mat73.loadmat('/content/Hamm_RE_n_15_100_10k_test-1.mat')
bt_1 = mat73.loadmat('/content/BCH_RE_n_15_100_10k_test-1.mat')

x_t,y_t=test_pre(ht_1,bt_1)

acc1=pred(mod1,x_t,y_t)
acc2=pred(mod2,x_t,y_t)
acc3=pred(mod3,x_t,y_t)
acc4=pred(mod4,x_t,y_t)
acc5=pred(mod5,x_t,y_t)
acc6=pred(mod6,x_t,y_t)
acc7=pred(mod7,x_t,y_t)
acc8=pred(mod8,x_t,y_t)
acc9=pred(mod9,x_t,y_t)
acc10=pred(mod10,x_t,y_t)

accuracy_1=pd.concat([ pd.DataFrame([acc1]),pd.DataFrame([acc2]),  pd.DataFrame([acc3]),pd.DataFrame([acc4]), pd.DataFrame([acc5]),pd.DataFrame([acc6]),pd.DataFrame([acc7]),
                      pd.DataFrame([acc8]), pd.DataFrame([acc9]), pd.DataFrame([acc10])])
print(accuracy_1)
