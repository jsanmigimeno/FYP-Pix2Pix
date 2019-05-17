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

def parseLogTrain(filePath, cleanData=True, plot=False, fileURL=None, downloadFile=False, printEpoch=None):
    if downloadFile:
        dgd(fileURL, filePath)

    try:
        # Column names of log file
        colNames = ["epochT", "epoch", "G_GANT", "G_GAN", "G_L1T", "G_L1", "G_DescT", "G_Desc", "G_MatchT", "G_Match", "D_realT", "D_real", "D_fakeT", "D_fake"]
        useCols = ['epoch', 'G_GAN', 'G_L1', 'G_Desc', 'G_Match', 'D_real', 'D_fake']
        # Import data
        data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False)
    except:
        # Column names of log file
        colNames = ["epochT", "epoch", "G_GANT", "G_GAN", "G_L1T", "G_L1", "D_realT", "D_real", "D_fakeT", "D_fake"]
        useCols = ['epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
        # Import data
        data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False)

    if cleanData:
        data = cleanDF(data, 'epoch')
        data = data.apply(pd.to_numeric)

    if printEpoch is not None:
        message = "Epoch %i: " %printEpoch
        printNames = useCols[1:]
        for name in printNames:
            message += "%s: %.4f " % (name, data[name][printEpoch-1])
        print(message)

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


def parseLogVal(filePath, version='auto', cleanData=True, plot=False, fileURL=None, downloadFile=False, printMin=False, printEpoch='Min'):
    if downloadFile:
        dgd(fileURL, filePath)

    # Column names of log file
    colNames = ["epochT", "epoch", "L1T", "L1", "PSNRT", "PSNR", "SSIMT", "SSIM", "DescT", "Desc", "MatchT", "Match"]
    useCols = ["epoch", "L1", "PSNR", "SSIM", "Desc", "Match"]
    # Import data
    data = pd.read_csv(filePath, sep=" ", header=None, names=colNames, usecols=useCols, skiprows=1, index_col=False, error_bad_lines=False)
    
    if cleanData:
        data = cleanDF(data, 'epoch')
        data = data.apply(pd.to_numeric)

    netLoss = data['L1'] + data['Desc']
    if printMin:
        minIdx = np.argmin(netLoss.values)
        print("Minimum at %i, %.4f (L1 + Desc)" % (minIdx, netLoss.values[minIdx]))

    if printEpoch is not None:
        if printEpoch == 'Min':
            printEpoch = minIdx + 1

        message = "Epoch %i: " %printEpoch
        printNames = useCols[1:]
        for name in printNames:
            message += "%s: %.4f " % (name, data[name][printEpoch-1])
        print(message)

    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()
        plt.title('Validation Losses')
        plotNames = useCols[1:]
        for name in plotNames:
            ax.plot(data[name].values, label=name)
        ax.plot(netLoss.values, label='L1 + Desc')
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
# valD = parseLogVal("./temp_logs/OnlyGAN.txt", plot=False, fileURL='https://drive.google.com/open?id=1-VNBmXsLBICbIKS0qulHtkpjfd7u9L_4', downloadFile=False, printMin=True)
# valD = parseLogTrain("./temp_logs/TrainOnlyGAN.txt", plot=False, fileURL='https://drive.google.com/open?id=1-KoSyT9WC1w-qYB37gE02_1Czqo1_ldM', downloadFile=False, printEpoch=160)

# Baseline - Only L1
# valD = parseLogVal("./temp_logs/OnlyL1.txt", plot=True, fileURL='https://drive.google.com/open?id=1-ZQVhkaWqQs_n6E6enRK-3ZLIx611RPP', downloadFile=False, printMin=True)
# valD = parseLogTrain("./temp_logs/TrainOnlyL1.txt", plot=False, fileURL='https://drive.google.com/open?id=1-WcP8L9zRXyL8ZJ2-w4Tz-_dSXcCshnB', downloadFile=False, printEpoch=175)

# Descriptor Loss - lambda 150
#valD = parseLogVal("./temp_logs/Desc150.txt", plot=False, fileURL='https://drive.google.com/open?id=1-S2wtRwWO5W8vKvq81SiwakMeC8ElfY_', downloadFile=False, printMin=True)
#valD = parseLogTrain("./temp_logs/TrainDesc150.txt", plot=False, fileURL='https://drive.google.com/open?id=1-Q7mgAA8jNMdn4GT2XlNm21AzOFdU5cE', downloadFile=False, printEpoch=186)

# Siamese Loss - desc lambda 100
# valD = parseLogVal("./temp_logs/Siamese100.txt", plot=False, fileURL='https://drive.google.com/open?id=1MWm-EExerW6Gc-CgtViBgivrYg9hnBJM', downloadFile=False, printMin=True)
# valD = parseLogTrain("./temp_logs/TrainSiamese100.txt", plot=False, fileURL='https://drive.google.com/open?id=1MSqFxxcRxaltEjG3jrxgOrJ1Ir9K4e8e', downloadFile=True, printEpoch=172)

# Siamese Loss - desc lambda 100 2
# valD = parseLogVal("./temp_logs/Siamese100_2.txt", plot=True, fileURL='https://drive.google.com/open?id=1Vpy6I1YNxgPrl8KnPcS30Y8qMkD3IcJ5', downloadFile=True, printMin=True)
# valD = parseLogTrain("./temp_logs/TrainSiamese100_2.txt", plot=False, fileURL='https://drive.google.com/open?id=1VitE4omOXdDX9Cm-FirPQ96n_OTFQ6Cs', downloadFile=True, printEpoch=172)

# Baseline
# valD = parseLogVal("./temp_logs/Baseline.txt", plot=False, fileURL='https://drive.google.com/open?id=10km2K9Tk-yRn6QLD5PBaWfyklpIG6f3p', downloadFile=True, printMin=True)
# valD = parseLogTrain("./temp_logs/TrainBaseline181.txt", plot=False, fileURL='https://drive.google.com/open?id=10VvU1PWz2uB4b5_QbXPrhrYlvARpvoEE', downloadFile=False, printEpoch=181)

# Descriptor Loss - lambda 100
# valD = parseLogVal("./temp_logs/Desc100.txt", plot=True, fileURL='https://drive.google.com/open?id=13SsTOzddTRRVoZTteec5IGAqkG71Fkd_', downloadFile=True, printMin=True)

# Descriptor Loss - lambda 100 per channel loss
#valD = parseLogVal("./temp_logs/Desc100C.txt", plot=True, fileURL='https://drive.google.com/open?id=12H1KcdfRvjFQm4-qQVAbiAePcNiWyDrl', downloadFile=True, printMin=True)

# Siamese Loss - lambda 100 per channel loss
valD = parseLogVal("./temp_logs/Siamese100C.txt", plot=True, fileURL='https://drive.google.com/open?id=1nY0hBOofl_XDJKa3IhhYdKHIOEBFVxHm', downloadFile=True, printMin=True)