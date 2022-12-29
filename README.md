## Implementation of paper - Detecting_Linear_Block_Codes_via_Deep_Learning


### Dataset Generation
we have generated dataset in a stepwise (for example N=15 No preprocessing  100,200,..1k,2k,...5k. for each type of codeword) and saved this stepwise dataset in a Matlab file(.matfile).Then we used .mat73 package to import this .matfiles to pandas dataframe in python to train the NN classifier



## Training

+ Run  Train_No_Data_Preprocessing.py python script to train the classification algorithm for N=15 length codes (No Data Preprocessing) 

+ Run Train_Data_processing_Hamming_distance_and_inner_product.py python script to train the classification algorithm for N=15 length codes(Data processing based on the Hamming distance & inner-product)

***

# Citation
```
@article{
  title = {Detecting Linear Block Codes via Deep Learning},
  author = {Arti Yardi (IIIT Hyderabad), Vamshi Krishna Kancharla (IIIT Bangalore), Amrita Mishra(IIIT Bangalore)},
  Conference = {IEEE WCNC 2023, (IEEE Wireless Communications and Networking Conference 2023)},
  year={2023}
  }
```
