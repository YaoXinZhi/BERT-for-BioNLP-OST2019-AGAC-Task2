# BERT-for-BioNLP-OST2019-AGAC-Task2
BERT-for-BioNLP-OST2019-AGAC-Task2

### How to Cite us ?  
Please cite follow work, if you use this code:  
Yuxing Wang, Kaiyin Zhou, Mina Gachloo, Jingbo Xia*.** An Overview of the Active Gene Annotation Corpus and the BioNLP OST 2019 AGAC Track Tasks.** BioNLP Open Shared Task 2019, workshop in EMNLP-IJCNLP 2019. Page: 62-71, Hong Kong, 2019.  


### Virtual Environment
You can build a virtual environment for project operation.  
```
# Building a virtual environment
pip3 install virtualenv
pip3 install virtualenvwrapper

virtualenv -p /usr/local/bin/python3.6 $env_name --clear  

# active venv.
source $env_name/bin/activate  

# deactive venv.
deactivate
```

### Requirements

```
pip3 install -r requirements.txt
```
If you cannot download torch automatically through requirements.txt, you can delete the torch version information and get the command line of torch installation from the [torch official website](https://pytorch.org/). Note that the installed torch version needs to be the same as that in requirenemts.txt.

**OSX**  
```
pip3 install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

**Linux and Windos**  
```
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```



### Default Run

**Model training and evaluation**
```
python3 main.py
```
**modify hyperparameters**  
You can modify the model hyperparameters by editing the config.py file.  
```vi config.py```



### Data
AGAC_answer.rar and AGAC_training.rar contain the original AGAC-Json files.  
The label.txt file contains three types of relationship labels, CauseOf, ThemeOf and NoRelation.  
train_relations.txt and test_relations.txt contain files for the corresponding statistics of the label-relationship in the training set and the test set, which are used to construct negative samples.  
The train.txt and test.txt files contain all the artificially annotated relationships and the constructed negative samples.
Please ignore the train.small.txt and test.small.txt files, I use these two files for debugging.  
```
data/AGAC_answer.rar
data/AGAC_training.rar
data/labels.txt
data/train_relations.txt
data/test_relations.txt
data/train.txt
data/test.txt
data/train.small.txt
data/test.small.txt
```





