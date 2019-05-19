import os
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
from random import randint
from math import ceil

dataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"
manDataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\baselineTestMeanAdjust"
runsDir = "E:\\FYP"

runNames = {
    'DSBaseline': 181,
    'DSBaselineOnlyGAN': 160,
    'DSBaselineOnlyL1': 175,
    'DSDescriptorLossOld': 180,
    'DSDescriptorLoss150': 186,
    'DSSiamese': 172,
    'DSSIFT100': 188,
    'DSGANDescriptor100': 34,
    'DSDescriptor100_2' : 117
}

baselineNames = ['DSBaseline', 'DSBaselineOnlyGAN', 'DSBaselineOnlyL1']
descriptorNames = ['DSBaseline', 'DSGANDescriptor100', 'DSDescriptor100_2', 'DSSIFT100']
setNames = ['baseline - 1', 'descriptor - 2', 'siamese - 3']

class Comparator():
    def __init__(self, dataDir, manDataDir, runsDir, runNames, ids=[]):
        self.dataDir = dataDir
        self.manDataDir = manDataDir
        self.runsDir = runsDir
        self.runNames = runNames
        self.ids = ids
        self.makeFigure()
    
    def makeFigure(self):
        self.nImages = len(self.runNames)

        nCols = 3
        nRows = ceil(self.nImages/nCols) + 1

        self.fig, self.axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(2.5*nCols, 2*nRows + 0.25), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.1)

        axnext = plt.axes([0.9, 0.01, 0.075, 0.075]) #left, bottom, width, height
        bnext = plt.Button(axnext, 'Next')
        bnext.on_clicked(self.plotImages)

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
                    isFile = os.path.isfile(os.path.join(manDataDir, f))
                
                imgId = os.path.splitext(f)[0]
            print("Image id: %s" % imgId)

        # Get (man) baseline image path
        imgPath = os.path.join(self.dataDir, imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        self.axs[0, 0].clear()
        self.axs[0, 0].imshow(img[:, 0:imgWidth])
        self.axs[0, 0].title.set_text('Man Baseline')

        imgPath = os.path.join(manDataDir, imgId + '.jpg')
        img = image.imread(imgPath)
        imgWidth = img.shape[1]//2
        imgHeight = img.shape[0]
        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(img[:, 0:imgWidth])
        
        self.axs[0, 1].title.set_text('Man Baseline')

        self.axs[0, 2].clear()
        self.axs[0, 2].imshow(img[:, imgWidth:])
        self.axs[0, 2].title.set_text('GT')

        axIdx = 0
        for name, epoch in self.runNames.items():
            imgPath = os.path.join(self.runsDir, name, 'test_' + str(epoch), 'images', imgId + '_fake_B.jpg')
            img = image.imread(imgPath)
            img = cv2.resize(img, (imgWidth, imgHeight))
            self.axs[axIdx//3+1, axIdx % 3].clear()
            self.axs[axIdx//3+1, axIdx % 3].imshow(img)
            self.axs[axIdx//3+1, axIdx % 3].title.set_text(name)
            axIdx += 1

        plt.axis('scaled')

        for ax in self.axs.flatten():
            ax.axis('off')

        self.fig.canvas.draw_idle()

message = "Select set: "
for name in setNames: 
    message += "%s " % name

selection = input(message)

if selection == '0':
    names = runNames
elif selection == '1':
    names = baselineNames
elif selection == '2':
    names = descriptorNames
else:
    names = runNames

runsPlot = dict((k, runNames[k]) for k in names)

ids = ['10003_00_0.1s', '10170_03_0.1s', '10032_03_0.1s', '10228_04_0.04s' ,'10006_06_0.1s', '10191_09_0.04s', '10016_09_0.1s', '10054_00_0.1s']
comp = Comparator(dataDir=dataDir, manDataDir=manDataDir, runsDir=runsDir, runNames=runsPlot, ids=ids)