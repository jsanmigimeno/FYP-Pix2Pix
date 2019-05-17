import os
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
from random import randint

dataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"
manDataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\baselineTestMeanAdjust"
runsDir = "E:\\FYP"

runNames = {
    'DSBaseline': 181,
    #'DSBaselineOnlyGAN': 160,
    #'DSBaselineOnlyL1': 175,
    'DSDescriptorLossOld': 180,
    'DSSiamese': 172
}

nImages = len(runNames)

if nImages < 3:
    nCols = 3
else:
    nCols = nImages

fig, axs = plt.subplots(nrows=2, ncols=nCols, figsize=(2.5*nCols, 4), sharex=True, sharey=True)

nextImg = True

while nextImg:
    #imgId = '10003_00_0.1s'
    imgId = ''

    if imgId == '':
        files = os.listdir(manDataDir)
        nFiles = len(files)
        isFile = False
        while not isFile:
            idx = randint(0, nFiles)
            f = files[idx]
            isFile = os.path.isfile(os.path.join(manDataDir, f))
        
        imgId = os.path.splitext(f)[0]
        print("Image id: %s" % imgId)

    # Get (man) baseline image path
    imgPath = os.path.join(dataDir, imgId + '.jpg')
    img = image.imread(imgPath)
    imgWidth = img.shape[1]//2
    imgHeight = img.shape[0]
    axs[0, 0].clear()
    axs[0, 0].imshow(img[:, 0:imgWidth])
    axs[0, 0].axis('off')
    axs[0, 0].title.set_text('Man Baseline')

    imgPath = os.path.join(manDataDir, imgId + '.jpg')
    img = image.imread(imgPath)
    imgWidth = img.shape[1]//2
    imgHeight = img.shape[0]
    axs[0, 1].clear()
    axs[0, 1].imshow(img[:, 0:imgWidth])
    axs[0, 1].axis('off')
    axs[0, 1].title.set_text('Man Baseline')

    axs[0, 2].clear()
    axs[0, 2].imshow(img[:, imgWidth:])
    axs[0, 2].axis('off')
    axs[0, 2].title.set_text('GT')

    axIdx = 0
    for name, epoch in runNames.items():
        imgPath = os.path.join(runsDir, name, 'test_' + str(epoch), 'images', imgId + '_fake_B.jpg')
        img = image.imread(imgPath)
        img = cv2.resize(img, (imgWidth, imgHeight))
        axs[1, axIdx].clear()
        axs[1, axIdx].imshow(img)
        axs[1, axIdx].axis('off')
        axs[1, axIdx].title.set_text(name)
        axIdx += 1

    plt.axis('scaled')
    plt.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.02)

    fig.canvas.draw_idle()
    plt.show(block=False)

    textInput = input("Press Enter for new image, any other key to exit: ")
    if textInput == "":
        print("Next image...")
    else:
        nextImg = False