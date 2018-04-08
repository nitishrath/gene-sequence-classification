#!/usr/bin/python
import random
import numpy as np
positive=random.sample(range(1000), 200)
negative=random.sample(range(1000), 200)
test=positive+[i+1000 for i in negative]
fi = open("train_processed.csv","r")
ftrain=open("train1.csv","w")
ftest=open("test1.csv","w")
seq=fi.read()
seq=seq.strip().split('\n')
for i in range(2000):
    if i in test:
        ftest.write(seq[i]+"\n")
    else: 
        ftrain.write(seq[i]+"\n")
ftrain.close()
ftest.close()
