import sys
import numpy as np

def roc_curve(y_true, y, step=0.01):

    pos_indices = np.where(y_true == 1)
    neg_indices = np.where(y_true == 0)

    thresholds = np.arange(0,1,step)
    
    fpr = np.zeros(thresholds.shape)
    tpr = np.zeros(thresholds.shape)

    for i,threshold in enumerate(thresholds):
        try:
            fpr[i] += float(np.where(y[neg_indices] >= threshold)[0].shape[0]) / neg_indices[0].shape[0]
        except ZeroDivisionError:
            fpr[i] += 1.0
        try:
            tpr[i] += float(np.where(y[pos_indices] >= threshold)[0].shape[0]) / pos_indices[0].shape[0]
        except ZeroDivisionError:
            tpr[i] += 0.0
    
    fpr = np.concatenate((fpr,np.array([0])))
    tpr = np.concatenate((tpr,np.array([0])))

    return fpr, tpr, thresholds
