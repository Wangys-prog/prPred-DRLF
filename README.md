## prPred-DRLF
prPred-DRLF is a tool to identify the plant resistance proteins (R proteins) based on deep representation learning features

prPred-DRLF is an open-source Python-based toolkit, which operates depending on the Python environment (##Python Version 3.7). 

### **If your computer has GPU,it will be faster.  **


### **Download**

> git clone git@github.com:Wangys-prog/prPred-DRLF.git


### **Install dependencies**

> pip3 install -r requirements.txt 

or 

> pip3 install joblib==1.0.1 
> 
> pip3 install tape_proteins==0.4
>  
> pip3 install numpy==1.19.2 
> 
> pip3 install pandas==1.2.0 
> 
> pip3 install Bio==0.4.1 
> 
> ## for python3.7 
> 
> ##if you have GPU  # CUDA 9.2  
> 
> pip install torch==1.2.0 torchvision==0.4.0 
> 
> #If you have GPU CUDA 10.0 
> 
> pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
> 
> ## CPU Only python3.7
> 
> pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
> 


### Input parameters

> python3 prPred-DRLF.py -h  
> $ usage: Script for predicting plant R protein using deep representation learning features  
> $       [-h] [-i I] [-o O]  
> $ optional arguments:  
> $  -h, --help  show this help message and exit  
> $  -i I        input sequences in Fasta format  
> $  -o O        path to saved CSV file  

> python3 prPred-DRLF.py -i ./dataset/test_data.fasta -o ./dataset/predict_result


### Webserver

>http://lab.malab.cn/soft/prPred-DRLF

### Other tools

>http://lab.malab.cn/~wys/

### Please cite
#### Yansu Wang, Lei Xu, Quan Zou, Chen Lin. prPred-DRLF: plant R protein predictor using deep representation learning features. Proteomics. 2021. DOI: 10.1002/pmic.202100161
#### Yansu Wang, Murong Zhou, Quan Zou, Lei Xu. Machine learning for phytopathology: from the molecular scale towards the network scale. Briefings in Bioinformatics. 2021, Doi: 10.1093/bib/bbab037
### Wang Y, Wang P, Guo Y, et al. prPred: A Predictor to Identify Plant Resistance Proteins by Incorporating k-Spaced Amino Acid (Group) Pairs[J]. Frontiers in bioengineering and biotechnology, 2021, 8: 1593.
