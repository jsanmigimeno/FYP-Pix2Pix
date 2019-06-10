#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from GDDownload import download_from_gd as dgd
from OneDownload import download_from_onedrive as dod

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

def parseLogVal(filePath, version='auto', cleanData=True, plot=False, fileURL=None, downloadFile=False, printMin=False, printEpoch='Min', title=None, save=False):
    if downloadFile:
        if 'google' in fileURL:
            dgd(fileURL, filePath)
        else:
            dod(fileURL, filePath)

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
        fig, ax = plt.subplots(figsize=(8, 3.5))
        plt.plot(data['L1'].values/2)
        if title is None:
            title = 'Validation Loss'
        plt.title(title, fontsize=18)
        plt.xlabel('Epoch', fontsize=16)
        #plt.xticks([0] + list(range(24, 224, 25)), [1] + list(range(25, 225, 25)), fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('L1', fontsize=16)
        plt.subplots_adjust(left=0.095, right=0.980, top=0.910, bottom=0.160)
        plt.grid()

        if save:
            fileName = './figures/' + title.replace(" ", "") + '.eps'
            plt.savefig(fileName)

        plt.show()

    return data

# Data
runData = {
    # name                      : (trainUrl, valUrl, completed)
    'DSBaseline'                : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EaDDBK7A0CFIroCXFtp2XO0BKanbYDkeMDWFuLSpbMlIwQ?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EeKn0BA3Ue5GtLFX031BPTABNZpg95AsvEnNteAf6SiMlg?download=1', False),
    'DSBaselineOnlyGAN'         : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/ERJk3E5dcn1HifvzPZUz72UB6RAWgmLpaRWaUqgf8WOIuA?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EZA2_jDoqFlAtTfWgyISzFMBmf8YzOmovRcDdIQuNGH3eg?download=1', False),
    'DSBaselineOnlyL1'          : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EZg7U5gF1TFMoSVDmEEw1pABiewA01sZX7OWhbZvcHwJfw?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EXucrF6sMh9ApdAETh4-nX0Bm1aIr0CXHcU68WE6_3lo7w?download=1', False), 
    #'DSDescriptorLoss100'       : ('', 'https://drive.google.com/open?id=13SsTOzddTRRVoZTteec5IGAqkG71Fkd_', False),
    'DSDescriptorLoss100_2'     : ('https://drive.google.com/open?id=1VitE4omOXdDX9Cm-FirPQ96n_OTFQ6Cs', 'https://drive.google.com/open?id=1Vpy6I1YNxgPrl8KnPcS30Y8qMkD3IcJ5', True),
    'DSDescriptorLoss100C'      : ('', 'https://drive.google.com/open?id=12H1KcdfRvjFQm4-qQVAbiAePcNiWyDrl', False),
    'DSDescriptorLoss150'       : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EWG7l1k5hn5MoGV3kV1yZCoBo4yk_BZOHnUZvA8VPehlMA?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EZli-b1DKr1FjE1ySAXKAKsB9WjKEUZVXAgBzAPSa4lSDQ?download=1', False),
    'DSSiamese'                 : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EYlvQXil-25HmVfmt9PpmOAB6aEwxQg1xXFAVSh4uT07QA?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/Ec_zEDO4VNlAsa5Q0ZFF-xABHs4svO4x-4URPIOC4xOrbw?download=1', False),
    'DSSiamese100C'             : ('https://drive.google.com/open?id=1nWAUYVkAn8PYEp5yLTHzpGI7BHYWx2JA', 'https://drive.google.com/open?id=1nY0hBOofl_XDJKa3IhhYdKHIOEBFVxHm', False),
    'DSSIFT100'                 : ('https://drive.google.com/open?id=19sTx4rkjVS6nrdyEU14CRgH0GrT-zpjx', 'https://drive.google.com/open?id=19ub6N2Yt9uuO8Yi-V5bAkt7QT0WSfm4j', True), 
    'DSGANDescriptor100'        : ('https://drive.google.com/open?id=1mmk4iLayLJcAZbjbm29NE2bIhgN-zPcj', 'https://drive.google.com/open?id=1mnSJo9hY88EEwMhalUW7OGzNuQwi3AKZ', False),
    'DSSiameseSIFT'             : ('https://drive.google.com/open?id=18a2OXLTA2DQsKgHI1LrnXJ6JI_B-jD3y', 'https://drive.google.com/open?id=18fXNlFF61-KEcOqKnRBtfzPrUsv4EBOn', True),
    'DSSiameseNoL1'             : ('https://drive.google.com/open?id=1-IbyStSC1qoWv2EoXWf0Z7w-UQrfavku', 'https://drive.google.com/open?id=1-JBxlWIF0CiXIQrOCJ0ZKXkww41oUEgx', False),
    'DSDescriptorNEP'           : ('', 'https://drive.google.com/open?id=1-kesyz3aYSB3tZ1wltwtTRWeYhPC6pzy', False),
    'DSDescriptor100C_2'        : ('https://drive.google.com/open?id=1-C9gyUm8nafsWK5y_r_S68UxtzEtNTZx', 'https://drive.google.com/open?id=1-ESjy4g5LfOj6NoMsOD-DkmE_oQ6X0LA', True),
    'DSDescriptor200'           : ('https://drive.google.com/open?id=106HwnLKKWreSs_0DqVTJUhg_osQN0von', 'https://drive.google.com/open?id=106T7donTpAblst_IkvYDc63Vl6n8z5El', False),
    'DSDescLog'                 : ('', 'https://drive.google.com/open?id=1-tQXYghd92ftv_R_Lmya4Nc0uDYukazq', False),
    'DSBaselineLog'             : ('', 'https://drive.google.com/open?id=1O--eM8DcO4kgImhV8Sre9EAU5K8fY4ew', False),
    'DSSiameseLog'              : ('', 'https://drive.google.com/open?id=1-RHFvZ1mApXMiK81gyjoWzisWgciyzi4', False),
    'DSSiameseLogC'             : ('', 'https://drive.google.com/open?id=1-97XBcyc_Y0zIvutToUNzSrMjDCsoMqG', False),
    'DSDescriptor300'           : ('https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EXsqHZwxSMxIpuSXJvGGvdYBrIaeTvbXPgGhW79VtSzWFA?download=1', 'https://imperiallondon-my.sharepoint.com/:t:/g/personal/js5415_ic_ac_uk/EdTWo_4q5cZMjWGGNluiWR8B91Bu7S8ytL5h_jEEL619Hw?download=1', False),
    'DSDescriptorLog200'        : ('', 'https://drive.google.com/open?id=1-RkaGYFX1BTB6247ierovsSwClroC-_O', False),
    'DSDescriptorLog300'        : ('', 'https://drive.google.com/open?id=1-IWtRAffu6WOUqD2W_yFSt6wSqPV50WD', False),
    'DSDescriptorLog400'        : ('', 'https://drive.google.com/open?id=1-LW3afFms0YxR5oiiWtfvq5sJJZ3cGcu', False),
    'DSDescriptorLog500'        : ('', 'https://drive.google.com/open?id=1-M-SHDZZVAXEYUvVTJBrWll0T9IkZGWY', False),
    'DSSiameseLog200'           : ('', 'https://drive.google.com/open?id=1-c3pt35rY7FzkHNPYszza177r3Um2cTz', False),
    'DSSiameseLog300'           : ('', 'https://drive.google.com/open?id=1-YboSiRveJfLwdAzp0XhIsTcf5hK891e', False),
    'DSSiameseLog400'           : ('', 'https://drive.google.com/open?id=1-NPl19SDd0vZU7Vi94hqYA_6WQCA-Mkm', False),
    'DSSiameseLog500'           : ('', 'https://drive.google.com/open?id=1-WO8ivF4v1ivoBvKo_iRshBCZDJ1QH6i', False),
}

forceDownload = False

nameId = 'DSSiameseLog500'
title = 'Baseline Validation Loss'
save = True
urls = runData[nameId]

valData = parseLogVal("./temp_logs/" + nameId + '_val.txt', plot=True, fileURL=urls[1], downloadFile=((not urls[2]) or forceDownload), printMin=True, save=save, title=title)
#trainData = parseLogTrain("./temp_logs/" + nameId + '_train.txt', plot=False, fileURL=urls[0], downloadFile=((not urls[2]) or forceDownload), printEpoch=None)

# nameIds = ['DSSIFT100', 'DSSiameseSIFT']
# for name in nameIds:
#     print(name)
#     urls = runData[name]
#     valData = parseLogVal("./temp_logs/" + name + '_val.txt', plot=False, fileURL=urls[1], downloadFile=((not urls[2]) or forceDownload), printMin=True)