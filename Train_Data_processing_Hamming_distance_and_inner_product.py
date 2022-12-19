import mat73
import pandas as pd
from third_fourth_prep import mat_f3,dat_samp,train,test_pre3,pred 

#Importing .matfies
dh_1 = mat73.loadmat('/content/Hamm_H-D_5_100_n(15)_P001_1.mat')
db_1 = mat73.loadmat('/content/BCH_H-D_5_100_n(15)_P001_1.mat')

dat1 = mat_f3(dh_1,db_1)

d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 = dat_samp(dat1)
d1.shape,d2.shape,d3.shape,d4.shape,d5.shape,d6.shape,d7.shape,d8.shape,d9.shape,d10.shape


e=10  ## epochs
b=1   ## batch_size

# Training 
mod1=train(d1,e,b)
mod2=train(d2,e,b)
mod3=train(d3,e,b)
mod4=train(d4,e,b)
mod5=train(d5,e,b)
mod6=train(d6,e,b)
mod7=train(d7,e,b)
mod8=train(d8,e,b)
mod9=train(d9,e,b)
mod10=train(d10,e,b)


##Testing

ht_1 = mat73.loadmat('/content/Hamm_H-D_5_100_n(15)_P001_test_1.mat')
bt_1 = mat73.loadmat('/content/BCH_H-D_5_100_n(15)_P001_test_1.mat')

x_t,y_t=test_pre3(ht_1,bt_1)

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
