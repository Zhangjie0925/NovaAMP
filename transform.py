import numpy as np
import sys
import torch

def data2array(dataset):
    inputlist=[]
    labellist=[]    
    with open(dataset,'r')as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip()
        #line=line.decode('utf-8','ignore')
        input=[]
        label=[]
        for i in line.split(',')[0:1000]:
            input.append((int(i)))
        inputlist.append(input)     
        for i in line.split(',')[-1]:
            label.append((int(i)))
            
        labellist.append(label)        
    inputarray=np.array(inputlist)
    labelarray=np.array(labellist)
                
    return(inputarray,labelarray)
#print(data2array('./testdata.format.len1000.txt'))


              

