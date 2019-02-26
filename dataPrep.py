#Erin Narwhal, 2/23/18, Challenge 2
from __future__ import division
import numpy as np
import matplotlib.pyplot as mplot
import collections
import os
import re
import math
import csv
import pandas as pd

#opens sentiment lexicon
data=open("files/reasons.tsv", "r")
data=data.read()
data=data.split("\n")

#creates dictionary for lexicon
holding = dict()

#splits the lexicon text file
for x in range(0, len(data)-1):
    data[x]=data[x].split("\t")

def add(list, i, values):
    if(i < len(list)):
        values.append(list[i])
        i += 1
        add(list, i, values)
    return values

def addToList(list, value):
    return list.append(value)

#loop to populate dictionary
for x in range(0, len(data)-1350):
    # print(data[x])
    if(int(data[x][2]) == 1):
        if(len(data[x]) == 3):
            holding[data[x][1]] = [0]
        else:
            holding[data[x][1]] = add(data[x], 3, [])

prepKey = []
prepVals = []
prep=[]
for key, value in holding.items():
    for x in value:
        prepVals.append(x)
    for x in range(0, len(value)):
        prepKey.append(key)

prep.append(prepKey)
prep.append(prepVals)

prep_data = pd.DataFrame(
    {'tweet': prepKey,
     'reason': prepVals,
    })

prep_data.to_csv('prepared.csv', sep='\t')
