import os, sys
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
from random import randint
from math import ceil
import numpy as np
import csv

dataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"
manDataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\baselineTestMeanAdjust"
runsDir = "C:\\CodingSSD\\TestResults"

runNames = {
    'DSBaseline': (181, 'Baseline'),
    'DSBaselineOnlyGAN': (160, 'Baseline only GAN'),
    'DSBaselineOnlyL1': (175, 'Baseline only L1'),
    #'DSDescriptorLossOld': (180, '+ HardNet'),
    'DSDescriptorLoss150': (186, '+ HardNet 150'),
    'DSSiamese': (172, '+ HardNet Siamese'),
    'DSSIFT100': (188, '+ SIFT Desc'),
    'DSGANDescriptor100': (34, '+ HardNet (no L1)'),
    'DSDescriptor100_2' : (117, '+ HardNet'),
    'DSSiameseNoL1': (187, '+ HardNet Siamese (no L1)'),
    'DSSiamese100C': (192, '+ HardNet Siamese (RGB)'),
    'DSSiameseSIFT': (145, '+ SIFT Siamese'),
    #'DSDescriptor200': (195, '+ HardNet 200'),
    #'DSDescriptorNEP': (142, '+ HardNet (no emp. pat)')
}

baselineNames = ['DSBaseline', 'DSBaselineOnlyGAN', 'DSBaselineOnlyL1']
descriptorNames = ['DSBaseline', 'DSSIFT100', 'DSDescriptor100_2', 'DSGANDescriptor100', 'DSDescriptorNEP']
descriptorCompNames = ['DSBaseline', 'DSDescriptor100_2', 'DSDescriptorLoss150', 'DSDescriptor200']
siameseNames = ['DSBaseline', 'DSDescriptor100_2', 'DSSiamese', 'DSSiameseNoL1', 'DSSiamese100C', 'DSSiameseSIFT']
setNames = [baselineNames, descriptorNames, siameseNames]

class Comparator():
    def __init__(self, dataDir, manDataDir, runsDir, runNames, ids=[], savePath='./figures', saveOnly=False, splitPath = './split.csv', maxWidth=1000):
        self.dataDir = dataDir
        self.manDataDir = manDataDir
        self.runsDir = runsDir
        self.runNames = runNames
        self.savePath = savePath
        self.maxWidth = maxWidth

        # load split
        self.splits = []
        self.splitPath = splitPath
        if splitPath is not None:
            with open(splitPath, 'r') as f:
                reader = csv.reader(f)
                for line in enumerate(reader):
                    self.splits.append(line[1])

        self.ids = list(ids.keys())
        self.idsData = ids.copy()
        if saveOnly:
            self.saveAll()
        else:
            self.makeFigure()
    
    def printSplit(self, imgId):
        found = False
        for i, line in enumerate(self.splits):
            if imgId + '.jpg' in line:
                found = True
                print("Id: %s\t is in split %i" % (imgId, i))
                self.currentSplitId = i
                break
        if not found:
            print("No split found for id %s!" % key)
            self.currentSplitId = None

    def makeFigure(self):
        self.nImages = len(self.runNames)

        nCols = 3
        nRows = ceil(self.nImages/nCols) + 1

        self.fig, self.axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(2.5*nCols, 2*nRows + 0.25), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.1)

        axnext = plt.axes([0.9, 0.01, 0.075, 0.075]) #left, bottom, width, height
        bnext = plt.Button(axnext, 'Next')
        bnext.on_clicked(self.plotImages)

        savebtn = plt.axes([0.75, 0.01, 0.075, 0.075]) #left, bottom, width, height
        bsavebtn = plt.Button(savebtn, 'Save')
        bsavebtn.on_clicked(self.savePlots)

        self.plotImages()
        plt.show()#block=False)
    
    def plotImages(self, event="", imgId=""):
        if imgId == '':
            if len(self.ids) > 0:
                imgId = self.ids[0]
                del self.ids[0]
            else:
                files = os.listdir(self.manDataDir)
                nFiles = len(files)
                isFile = False
                while not isFile:
                    idx = randint(0, nFiles)
                    f = files[idx]
                    isFile = os.path.isfile(os.path.join(self.manDataDir, f))
                
                imgId = os.path.splitext(f)[0]
            self.printSplit(imgId)

        self.imgId = imgId

        # Get (man) baseline image path
        imgPath = os.path.join(self.dataDir, imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.axs[0, 0].clear()
        self.axs[0, 0].imshow(img[:, 0:imgWidth])
        self.axs[0, 0].title.set_text('Original')
        self.aspect = imgHeight/imgWidth

        imgPath = os.path.join(self.manDataDir, imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(img[:, 0:imgWidth])
        self.axs[0, 1].title.set_text('Man. Adjust')
        

        self.axs[0, 2].clear()
        self.axs[0, 2].imshow(img[:, imgWidth:])
        self.axs[0, 2].title.set_text('Ground Truth')

        axIdx = 0
        for name, (epoch, title) in self.runNames.items():
            imgPath = os.path.join(self.runsDir, name, 'test_' + str(epoch), 'images', imgId + '_fake_B.jpg')
            img = image.imread(imgPath)
            img = cv2.resize(img, (imgWidth, imgHeight))
            self.axs[axIdx//3+1, axIdx % 3].clear()
            self.axs[axIdx//3+1, axIdx % 3].imshow(img)
            self.axs[axIdx//3+1, axIdx % 3].title.set_text(title)

            axIdx += 1

        plt.axis('scaled')

        for ax in self.axs.flatten():
            ax.axis('off')

        self.fig.canvas.draw_idle()

    def savePlots(self, event, coords=[]):
        if coords==[]:
            xlim = self.axs[0,0].get_xlim()
            ylim = self.axs[0,0].get_ylim()
            
        else:
            xlim = (coords[0], coords[1])
            ylim = (coords[2], coords[3])

        print("Saving figure with coords: %i, %i, %i, %i" % (xlim[0], xlim[1], ylim[0], ylim[1]))
        xWidth = xlim[1]-xlim[0]
        yCenter = int((ylim[0]+ylim[1])/2)

        try:
            folderPath = os.path.join(self.savePath, self.imgId + '_' + str(self.currentSplitId))
            os.mkdir(folderPath)
        except:
            pass

        # Get (man) baseline image path
        imgPath = os.path.join(self.dataDir, self.imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        imgA = img[:, 0:imgWidth]

        yHeightHalf = int(xWidth*imgHeight/imgWidth/2) 

        imgCropped = imgA[int(yCenter-yHeightHalf):int(yCenter+yHeightHalf),int(xlim[0]):int(xlim[1])]
        self.saveImage(imgCropped, os.path.join(folderPath, 'Dark_' + str(self.currentSplitId) + '.jpg'))

        imgB = img[:, imgWidth:]
        imgCropped = imgB[int(yCenter-yHeightHalf):int(yCenter+yHeightHalf),int(xlim[0]):int(xlim[1])]
        self.saveImage(imgCropped, os.path.join(folderPath, 'GT_' + str(self.currentSplitId) + '.jpg'))

        imgPath = os.path.join(self.manDataDir, self.imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        imgA = img[:, 0:imgWidth]
        imgCropped = imgA[int(yCenter-yHeightHalf):int(yCenter+yHeightHalf),int(xlim[0]):int(xlim[1])]
        self.saveImage(imgCropped, os.path.join(folderPath, 'Man_' + str(self.currentSplitId) + '.jpg'))

        for name, (epoch, title) in self.runNames.items():
            title = title.replace(" ", "")
            imgPath = os.path.join(self.runsDir, name, 'test_' + str(epoch), 'images', self.imgId + '_fake_B.jpg')
            img = image.imread(imgPath)
            img = cv2.resize(img, (imgWidth, imgHeight))
            imgCropped = img[int(yCenter-yHeightHalf):int(yCenter+yHeightHalf),int(xlim[0]):int(xlim[1])]
            self.saveImage(imgCropped, os.path.join(folderPath, title + '_' + str(self.currentSplitId) + '.jpg'))
    
    def saveImage(self, image, path):
        if image.shape[1] > self.maxWidth:
            image = cv2.resize(image, (self.maxWidth, int(self.maxWidth*image.shape[0]/image.shape[1]) ))
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def saveAll(self):
        for imgId, coords in self.idsData.items():
            if coords is not None:
                self.imgId = imgId
                self.printSplit(imgId)
                self.savePlots(None, coords)
            

message = "Select set: All (0), Baseline (1), Descriptor (2), Siamese (3): "
selection = input(message)

ids = ['10003_00_0.1s', '10032_03_0.1s', '10228_04_0.04s' ,'10006_06_0.1s', '10191_09_0.04s', '10016_09_0.1s', '10054_00_0.1s'] #, '10170_03_0.1s'
ids = ['10003_00_0.1s', '10003_00_0.04s', '10016_00_0.1s', '10016_00_0.04s', '10030_00_0.1s', '10030_00_0.04s', '10032_00_0.1s', '10032_00_0.04s', '10040_00_0.1s', '10040_00_0.04s', '10074_00_0.1s', '10167_00_0.1s']

ids = {
    # '10003_00_0.1s'     : [926, 1257, 986, 761],
    # '10003_00_0.04s'    : [926, 1257, 986, 761],
    # '10016_00_0.1s'     : [144, 1125, 1248, 684],
    # '10016_00_0.04s'    : [144, 1125, 1248, 684],
    # '10030_00_0.1s'     : [0, 2119, 1415, 0],
    # '10030_00_0.04s'    : [0, 2119, 1415, 0],
    # '10032_00_0.1s'     : [1358, 1899, 1145, 858],
    # '10032_00_0.04s'    : [1358, 1899, 1145, 858],
    # '10040_00_0.1s'     : [449, 1531, 702, 40],
    # '10040_00_0.04s'    : [449, 1531, 702, 40],
    # '10074_00_0.1s'     : [0, 2119, 1415, 0],
    # '10167_00_0.1s'     : None,
    # '10228_04_0.04s'     : [205, 1826, 1165, 195],
    # '10006_06_0.1s'     : [292, 1379, 893, 224],
    # '10191_09_0.04s'     : [438, 1681, 908, 209],
    '10016_09_0.1s'     : None,
    '10054_00_0.1s'     : [706, 1988, 898, 85]
}

if selection == '0':
    for nameList in setNames:
        runsPlot = dict((k, runNames[k]) for k in nameList)
        comp = Comparator(dataDir=dataDir, manDataDir=manDataDir, runsDir=runsDir, runNames=runsPlot, ids=ids)
elif selection == '-1':
    comp = Comparator(dataDir=dataDir, manDataDir=manDataDir, runsDir=runsDir, runNames=runNames, ids=ids, saveOnly=True)
else:
    if selection == '1':
        names = baselineNames
    elif selection == '2':
        names = descriptorNames
    else:
        print("Selection not found!")
        sys.exit()
    runsPlot = dict((k, runNames[k]) for k in names)
    comp = Comparator(dataDir=dataDir, manDataDir=manDataDir, runsDir=runsDir, runNames=runsPlot, ids=ids)



