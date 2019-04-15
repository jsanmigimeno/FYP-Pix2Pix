#%%
import os
from matplotlib import pyplot, image
import numpy as np

path = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"
outPath = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\testMeanAdjust"

for filename in os.listdir(path):
    fullPath = os.path.join(path, filename)
    img = image.imread(fullPath)
    height = len(img)
    width = len(img[0])
    middle = int(width/2)
    shortExpImg = img[:, 0:middle]
    longExpImg = img[:, middle:]
    c = 255/np.log(1+np.max(shortExpImg))
    scaledImg = c*np.log(1+shortExpImg)
    # shortExpImg = img[:, 0:middle]/255
    # longExpImg = img[:, middle:]
    # scaleFactor = np.mean(longExpImg)/255/np.mean(shortExpImg)
    # scaledImg = shortExpImg*scaleFactor
    # scaledImg[scaledImg>1]=1
    out = np.concatenate((scaledImg, longExpImg), axis=1)


    savePath = os.path.join(outPath, filename)
    image.imsave(savePath, out.astype('uint8'))
    pass