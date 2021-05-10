## prPred-DRLF
prPred-DRLF is a tool to identify the plant resistance proteins (R proteins) based on deep representation learning features

prPred-DRLF is an open-source Python-based toolkit, which operates depending on the Python environment (Python Version 3.0 or above). 

### **If your computer has GPU,it will be faster. **


### **Download**

> git clone git@github.com:Wangys-prog/prPred-DRLF.git


### **Install dependencies**

> pip3 install -r requirements.txt


### Input parameters

> python3 prPred-DRLF.py -h  
> $ usage: Script for predicting plant R protein using deep representation learning features  
> $       [-h] [-i I] [-o O]  
> $ optional arguments:  
> $  -h, --help  show this help message and exit  
> $  -i I        input sequences in Fasta format  
> $  -o O        path to saved CSV file  

> python3 prPred-DRLF.py -i ./dataset/test_data.fasta -o ./dataset/predict_result
