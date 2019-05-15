#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from GDDownload import download_from_gd as dgd
#%%

def parseLog(filePath, version='auto', plot=False):
    if version == 'auto':
        with open(filePath, 'r') as f:
            _ = f.readline()
            line = f.readline()
            if 'G_Match' in line:
                version = 3
            elif 'G_Desc' in line:
                version = 2
            else:
                version = 1
    if version == 1:
        # Column names of log file
        colNames = ["epochT", "epoch", "itersT", "iters", "timeT", "time", "dataT", "data", "G_GANT", "G_GAN", "G_L1T", "G_L1", "D_realT", "D_real", "D_fakeT", "D_fake"]
        # Columns to import
        useCols = ['epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
    elif version == 2:
        # Column names of log file
        colNames = ["epochT", "epoch", "itersT", "iters", "timeT", "time", "dataT", "data", "G_GANT", "G_GAN", "G_L1T", "G_L1", "G_DescT", "G_Desc", "D_realT", "D_real", "D_fakeT", "D_fake"]
        # Columns to import
        useCols = ['epoch', 'G_GAN', 'G_L1', 'G_Desc', 'D_real', 'D_fake']
    elif version == 3:
        # Column names of log file
        colNames = ["epochT", "epoch", "itersT", "iters", "timeT", "time", "dataT", "data", "G_GANT", "G_GAN", "G_L1T", "G_L1", "G_DescT", "G_Desc", "G_MatchT", "G_Match", "D_realT", "D_real", "D_fakeT", "D_fake"]
        # Columns to import
        useCols = ['epoch', 'G_GAN', 'G_L1', 'G_Desc', 'G_Match', 'D_real', 'D_fake']
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False)
    # Remove comas in epoch column
    data["epoch"] = data["epoch"].str.replace(",","").astype(float)

    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()
        plt.title('Training Losses')
        plotNames = useCols[1:]
        for name in plotNames:
            ax.plot(data[name], label=name)
        ax.legend()
        plt.show()
    return data

def parseLogTrain(filePath, plot=False):
    # Column names of log file
    colNames = ["epochT", "epoch", "G_GANT", "G_GAN", "G_L1T", "G_L1", "G_DescT", "G_Desc", "G_MatchT", "G_Match", "D_realT", "D_real", "D_fakeT", "D_fake"]
    useCols = ['epoch', 'G_GAN', 'G_L1', 'G_Desc', 'D_real', 'D_fake']
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False)

    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()
        plt.title('Training Losses')
        plotNames = useCols[1:]
        for name in plotNames:
            ax.plot(data[name], label=name)
        ax.plot(data['G_GAN'] + data['G_L1'] + data['G_Desc'], label='G_Loss')
        ax.legend()
        plt.show()
    return data

def parseLogValOld(filePath):
    # Column names of log file
    colNames = ["epochT", "epoch", "valT", "val"]
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, skiprows=1, index_col=False)
    return data

def cleanDF(data, colName):
    noErrorFound = False
    while not noErrorFound:
        currentMax = 0
        errorIdx = None
        errorVal = None

        for i, e in enumerate(data[colName]):
            try:
                if float(e) > currentMax:
                    currentMax = float(e)
                else:
                    errorIdx = i
                    errorVal = float(e)
                    break
            except:
                continue
        
        if errorIdx is None:
            noErrorFound = True
        
        if not noErrorFound:
            for i, e in enumerate(data[colName].values):
                try:
                    if float(e) == errorVal:
                        data.drop(data.index[i:errorIdx], inplace=True)
                        break
                except:
                    continue
        
        return data


def parseLogVal(filePath, version='auto', cleanData=True, plot=False, fileURL=None, downloadFile=False):
    if downloadFile:
        dgd(fileURL, filePath)

    # Column names of log file
    colNames = ["epochT", "epoch", "L1T", "L1", "PSNRT", "PSNR", "SSIMT", "SSIM", "DescT", "Desc", "MatchT", "Match"]
    useCols = ["epoch", "L1", "Desc", "Match"]
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False, error_bad_lines=False)
    
    if cleanData:
        data = cleanDF(data, 'epoch')
        data = data.apply(pd.to_numeric)

    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()
        plt.title('Validation Losses')
        plotNames = useCols[1:]
        for name in plotNames:
            ax.plot(data[name].values, label=name)
        ax.plot((data['L1'] + data['Desc']).values, label='L1 + Desc')
        ax.legend()
        plt.show()
    return data

#logDf = parseLog("E:\FYP\Training Downscaled JPEG\loss_logMerged.txt")

# # # Plot standard log
# L = parseLog("C:\\Users\\Work\\Downloads\\loss_log (5).txt", plot=True)
# L1 = L['G_L1'].values
# Desc = L['G_Desc'].values
# epochs = L['epoch'].values.astype('int')
# maxEpoch = np.max(epochs)
# avgL1 = np.zeros(maxEpoch)
# avgDesc = np.zeros(maxEpoch)
# for e in range(1, maxEpoch):
#     avgL1[e] = np.mean(L1[epochs==e])
#     avgDesc[e] = np.mean(Desc[epochs==e])
# plt.plot(avgL1)
# plt.plot(avgDesc)
# plt.title('Descriptor Loss')
# plt.show()

# Parse train log
#parseLogTrain("C:\\Users\\Work\\Downloads\\train_loss_log.txt", plot=True)

# # # Plot validation old
# #valD = parseLogValOld("C:\\Users\\Work\\Downloads\\val_loss_log (3).txt")
# #valD = parseLogValOld("E:\\FYP\\Descriptor Loss\\val_loss_log.txt")
# # plt.plot(valD['val'].values)
# # plt.title('Descriptor Val. Loss')
# # plt.show()

# # Plot validation
#valD = parseLogVal("C:\\Users\\Work\\Downloads\\val_loss_log (8).txt", plot=True)

# Baseline - Only GAN
#valD = parseLogVal("./temp_logs/OnlyGAN.txt", plot=True, fileURL='https://drive.google.com/open?id=1-VNBmXsLBICbIKS0qulHtkpjfd7u9L_4', downloadFile=True)

# Baseline - Only L1
valD = parseLogVal("./temp_logs/OnlyL1.txt", plot=True, fileURL='https://drive.google.com/open?id=1-ZQVhkaWqQs_n6E6enRK-3ZLIx611RPP', downloadFile=True)

# Descriptor Loss - lambda 150
#valD = parseLogVal("./temp_logs/Desc150.txt", plot=True, fileURL='https://drive.google.com/open?id=1-S2wtRwWO5W8vKvq81SiwakMeC8ElfY_', downloadFile=True)

# Siamese Loss - desc lambda 100
#valD = parseLogVal("./temp_logs/Siamese100.txt", plot=True, fileURL='https://drive.google.com/open?id=1MWm-EExerW6Gc-CgtViBgivrYg9hnBJM', downloadFile=False)