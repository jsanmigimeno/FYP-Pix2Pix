#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
def parseLog(filePath):
    # Column names of log file
    colNames = ["epochT", "epoch", "itersT", "iters", "timeT", "time", "dataT", "data", "G_GANT", "G_GAN", "G_L1T", "G_L1", "D_realT", "D_real", "D_fakeT", "D_fake"]
    # Columns to import
    useCols = ['epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False)
    # Remove comas in epoch column
    data["epoch"] = data["epoch"].str.replace(",","").astype(float)
    return data

def parseLogVal(filePath):
    # Column names of log file
    colNames = ["epochT", "epoch", "valT", "val"]
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, skiprows=1, index_col=False)
    return data

#logDf = parseLog("E:\FYP\Training Downscaled JPEG\loss_logMerged.txt")

# Plot standard log
L = parseLog("C:\\Users\\Work\\Downloads\\loss_log (1).txt")
L1 = L['G_L1'].values
epochs = L['epoch'].values.astype('int')
maxEpoch = np.max(epochs)
avgL1 = np.zeros(maxEpoch)
for e in range(1, maxEpoch):
    avgL1[e] = np.mean(L1[epochs==e])
plt.plot(avgL1)
plt.show()

# Plot validation
valD = parseLogVal("C:\\Users\\Work\\Downloads\\val_loss_log (1).txt")
plt.plot(valD['val'].values)
plt.show()