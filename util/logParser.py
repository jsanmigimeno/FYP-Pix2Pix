#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from GDDownload import download_from_gd as dgd

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
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    cleanData = pd.DataFrame(columns=data.columns)

    for i, e in enumerate(data[colName]):
        if is_number(e):
            try:
                cleanData.iloc[int(float(e))-1] = data.iloc[i]
            except:
                cleanData = cleanData.append(data.iloc[i])
    
    cleanData.reset_index(drop=True, inplace=True)
        
    return cleanData

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
    minIdx = np.argmin(netLoss.values)

    if printMin:
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


# Data
runData = {
    # name                      : (trainUrl, valUrl, completed)
    'DSBaseline'                : ('https://drive.google.com/open?id=10VvU1PWz2uB4b5_QbXPrhrYlvARpvoEE', 'https://drive.google.com/open?id=10km2K9Tk-yRn6QLD5PBaWfyklpIG6f3p', False),
    'DSBaselineOnlyGAN'         : ('https://drive.google.com/open?id=1-KoSyT9WC1w-qYB37gE02_1Czqo1_ldM', 'https://drive.google.com/open?id=1-VNBmXsLBICbIKS0qulHtkpjfd7u9L_4', False),
    'DSBaselineOnlyL1'          : ('https://drive.google.com/open?id=1-WcP8L9zRXyL8ZJ2-w4Tz-_dSXcCshnB', 'https://drive.google.com/open?id=1-ZQVhkaWqQs_n6E6enRK-3ZLIx611RPP', False), 
    'DSDescriptorLoss100'       : ('', 'https://drive.google.com/open?id=13SsTOzddTRRVoZTteec5IGAqkG71Fkd_', False),
    'DSDescriptorLoss100_2'     : ('https://drive.google.com/open?id=1VitE4omOXdDX9Cm-FirPQ96n_OTFQ6Cs', 'https://drive.google.com/open?id=1Vpy6I1YNxgPrl8KnPcS30Y8qMkD3IcJ5', True),
    'DSDescriptorLoss100C'      : ('', 'https://drive.google.com/open?id=12H1KcdfRvjFQm4-qQVAbiAePcNiWyDrl', False),
    'DSDescriptorLoss150'       : ('https://drive.google.com/open?id=1-Q7mgAA8jNMdn4GT2XlNm21AzOFdU5cE', 'https://drive.google.com/open?id=1-S2wtRwWO5W8vKvq81SiwakMeC8ElfY_', False),
    'DSSiamese'                 : ('https://drive.google.com/open?id=1MSqFxxcRxaltEjG3jrxgOrJ1Ir9K4e8e', 'https://drive.google.com/open?id=1MWm-EExerW6Gc-CgtViBgivrYg9hnBJM', False),
    'DSSiamese100C'             : ('https://drive.google.com/open?id=1nWAUYVkAn8PYEp5yLTHzpGI7BHYWx2JA', 'https://drive.google.com/open?id=1nY0hBOofl_XDJKa3IhhYdKHIOEBFVxHm', False),
    'DSSIFT100'                 : ('https://drive.google.com/open?id=19sTx4rkjVS6nrdyEU14CRgH0GrT-zpjx', 'https://drive.google.com/open?id=19ub6N2Yt9uuO8Yi-V5bAkt7QT0WSfm4j', True), 
    'DSGANDescriptor100'        : ('https://drive.google.com/open?id=1mmk4iLayLJcAZbjbm29NE2bIhgN-zPcj', 'https://drive.google.com/open?id=1mnSJo9hY88EEwMhalUW7OGzNuQwi3AKZ', False),
    'DSSiameseSIFT'             : ('https://drive.google.com/open?id=18a2OXLTA2DQsKgHI1LrnXJ6JI_B-jD3y', 'https://drive.google.com/open?id=18fXNlFF61-KEcOqKnRBtfzPrUsv4EBOn', True),
    'DSSiameseNoL1'             : ('https://drive.google.com/open?id=1-IbyStSC1qoWv2EoXWf0Z7w-UQrfavku', 'https://drive.google.com/open?id=1-JBxlWIF0CiXIQrOCJ0ZKXkww41oUEgx', False)
}

forceDownload = False

nameId = 'DSSiamese100C'
urls = runData[nameId]

valData = parseLogVal("./temp_logs/" + nameId + '_val.txt', plot=True, fileURL=urls[1], downloadFile=((not urls[2]) or forceDownload), printMin=True)
#trainData = parseLogTrain("./temp_logs/" + nameId + '_train.txt', plot=False, fileURL=urls[0], downloadFile=((not urls[2]) or forceDownload), printEpoch=None)

# nameIds = ['DSSIFT100', 'DSSiameseSIFT']
# for name in nameIds:
#     print(name)
#     urls = runData[name]
#     valData = parseLogVal("./temp_logs/" + name + '_val.txt', plot=False, fileURL=urls[1], downloadFile=((not urls[2]) or forceDownload), printMin=True)