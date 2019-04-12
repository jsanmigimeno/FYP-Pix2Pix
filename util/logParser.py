#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
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

logDf = parseLog("E:\FYP\Training Downscaled JPEG\loss_logMerged.txt")