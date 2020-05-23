# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def entropy (labels):
    #Calculates entropy for two class table.
    n = len(labels)
    numFirstClass = np.sum((labels == 1))
    if (numFirstClass == 0 or numFirstClass == n):
        entropy = 0
    else:
        probFirstClass = numFirstClass / n
        probSecondClass = 1 - probFirstClass
        entropy = -(probFirstClass*np.log2(probFirstClass) + probSecondClass*np.log2(probSecondClass))
    return entropy


