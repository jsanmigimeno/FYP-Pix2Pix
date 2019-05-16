import os
from matplotlib import pyplot as plt
from matplotlib import image
import cv2

manDataDir = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\baselineTestMeanAdjust"
runsDir = "E:\\FYP"

runNames = {
    'DSBaseline': 181,
    #'DSBaselineOnlyGAN': 160,
    #'DSBaselineOnlyL1': 175,
    'DSDescriptorLossOld': 180,
    'DSSiamese': 172
}

imgId = '10003_00_0.1s'

nImages = len(runNames) + 2

fig, axs = plt.subplots(nrows=1, ncols=nImages, figsize=(2.5*nImages, 2), sharex=True, sharey=True)

# Get man baseline image path
imgPath = os.path.join(manDataDir, imgId + '.jpg')
img = image.imread(imgPath)
imgWidth = img.shape[1]//2
imgHeight = img.shape[0]
axs[0].imshow(img[:, 0:imgWidth])
axs[0].axis('off')
axs[0].title.set_text('Man Baseline')

axs[nImages-1].imshow(img[:, imgWidth:])
axs[nImages-1].axis('off')
axs[nImages-1].title.set_text('GT')

axIdx = 1
for name, epoch in runNames.items():
    imgPath = os.path.join(runsDir, name, 'test_' + str(epoch), 'images', imgId + '_fake_B.jpg')
    img = image.imread(imgPath)
    img = cv2.resize(img, (imgWidth, imgHeight))
    axs[axIdx].imshow(img)
    axs[axIdx].axis('off')
    axs[axIdx].title.set_text(name)
    axIdx += 1
    
plt.axis('scaled')
plt.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.02)
plt.show()